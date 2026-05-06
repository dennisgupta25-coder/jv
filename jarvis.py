"""
JARVIS-Local — Single-file voice assistant with OpenRouter multi-agent pipeline.
Wake word (local) → Record → Whisper (local STT) → Planner → Critic → Executor → Terminal.
"""
import json
import logging
import os
import signal
import sys
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np
import requests
import sounddevice as sd
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

colorama_init(autoreset=True)
load_dotenv()

# ════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════════

def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _get_int(key: str, default: int) -> int:
    v = _get(key)
    return int(v) if v else default

def _get_float(key: str, default: float) -> float:
    v = _get(key)
    return float(v) if v else default

def _get_bool(key: str, default: bool) -> bool:
    v = _get(key).lower()
    return v in ("1", "true", "yes", "y") if v else default


@dataclass
class Settings:
    # OpenRouter
    openrouter_api_key: str = field(default_factory=lambda: _get("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(default_factory=lambda: _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    http_referer: str = field(default_factory=lambda: _get("OPENROUTER_HTTP_REFERER", "http://localhost"))
    app_title: str = field(default_factory=lambda: _get("OPENROUTER_APP_TITLE", "JARVIS-Local"))

    # Configurable model IDs
    planner_model: str = field(default_factory=lambda: _get("PLANNER_MODEL", "openai/gpt-4o"))
    critic_model: str = field(default_factory=lambda: _get("CRITIC_MODEL", "google/gemini-pro-1.5"))
    executor_model: str = field(default_factory=lambda: _get("EXECUTOR_MODEL", "anthropic/claude-opus-4"))

    # Whisper
    whisper_size: str = field(default_factory=lambda: _get("WHISPER_MODEL_SIZE", "base"))
    whisper_device: str = field(default_factory=lambda: _get("WHISPER_DEVICE", "cpu"))
    whisper_compute: str = field(default_factory=lambda: _get("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_language: str = field(default_factory=lambda: _get("WHISPER_LANGUAGE", "en"))

    # Wake word
    wake_word: str = field(default_factory=lambda: _get("WAKE_WORD", "hey_jarvis"))
    wake_threshold: float = field(default_factory=lambda: _get_float("WAKE_WORD_THRESHOLD", 0.5))

    # Audio
    mic_device_index: int | None = None
    sample_rate: int = field(default_factory=lambda: _get_int("SAMPLE_RATE", 16000))
    silence_threshold: float = field(default_factory=lambda: _get_float("SILENCE_THRESHOLD", 0.01))
    silence_duration: float = field(default_factory=lambda: _get_float("SILENCE_DURATION_SEC", 1.5))
    max_record_sec: int = field(default_factory=lambda: _get_int("MAX_RECORD_SECONDS", 15))

    # Logging / memory
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: _get("LOG_FILE", "logs/jarvis.log"))
    enable_history: bool = field(default_factory=lambda: _get_bool("ENABLE_HISTORY", True))
    history_file: str = field(default_factory=lambda: _get("HISTORY_FILE", "memory/history.jsonl"))

    def __post_init__(self):
        idx = _get("MIC_DEVICE_INDEX")
        self.mic_device_index = int(idx) if idx else None
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing — copy .env.example to .env and fill it.")


settings = Settings()

# ════════════════════════════════════════════════════════════════════
# 2. LOGGER
# ════════════════════════════════════════════════════════════════════

def get_logger(name: str = "jarvis") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    fh = RotatingFileHandler(settings.log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


log = get_logger("jarvis")

# ════════════════════════════════════════════════════════════════════
# 3. TERMINAL OUTPUT (pretty colored, no TTS)
# ════════════════════════════════════════════════════════════════════

def banner(text: str) -> None:
    print(f"\n{Fore.CYAN}{'═' * 70}\n  {text}\n{'═' * 70}{Style.RESET_ALL}")

def status(text: str) -> None:
    print(f"{Fore.YELLOW}● {text}{Style.RESET_ALL}")

def agent_header(name: str, model: str) -> None:
    print(f"\n{Fore.MAGENTA}┌─ Agent: {name}  ({model}){Style.RESET_ALL}")

def agent_summary(text: str) -> None:
    print(f"{Fore.MAGENTA}│{Style.RESET_ALL} {text}")

def final_answer(text: str) -> None:
    print(f"\n{Fore.GREEN}╔══ FINAL ANSWER ══════════════════════════════════════════════════╗")
    print(f"{Fore.GREEN}{text}")
    print(f"{Fore.GREEN}╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

def user_text(text: str) -> None:
    print(f"{Fore.BLUE}🗣  You said: {Style.BRIGHT}{text}{Style.RESET_ALL}")

def err(text: str) -> None:
    print(f"{Fore.RED}✖ {text}{Style.RESET_ALL}")

# ════════════════════════════════════════════════════════════════════
# 4. WAKE-WORD LISTENER (openWakeWord, local)
# ════════════════════════════════════════════════════════════════════

CHUNK = 1280  # 80 ms @ 16 kHz — required by openWakeWord


class WakeWordListener:
    def __init__(self):
        self.model = WakeModel(wakeword_models=[settings.wake_word], inference_framework="onnx")
        self.threshold = settings.wake_threshold
        log.info(f"Wake-word model loaded: {settings.wake_word}")

    def listen(self) -> None:
        status(f"Listening for wake word '{settings.wake_word}' …")
        with sd.InputStream(
            samplerate=settings.sample_rate, channels=1, dtype="int16",
            blocksize=CHUNK, device=settings.mic_device_index,
        ) as stream:
            while True:
                audio, _ = stream.read(CHUNK)
                pcm = np.frombuffer(audio, dtype=np.int16)
                scores = self.model.predict(pcm)
                for kw, score in scores.items():
                    if score >= self.threshold:
                        log.info(f"Wake word detected ({kw}={score:.2f})")
                        status(f"✓ Wake word detected ({score:.2f})")
                        self.model.reset()
                        return

# ════════════════════════════════════════════════════════════════════
# 5. COMMAND RECORDER (silence-stop)
# ════════════════════════════════════════════════════════════════════

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def record_command() -> str:
    status("Recording your command (speak now)…")
    sr = settings.sample_rate
    block = int(sr * 0.1)
    silent_blocks_needed = int(settings.silence_duration / 0.1)

    frames: list[np.ndarray] = []
    silent_count = 0
    start = time.time()

    with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                        blocksize=block, device=settings.mic_device_index) as stream:
        while True:
            data, _ = stream.read(block)
            frames.append(data.copy())
            rms = np.sqrt(np.mean((data.astype(np.float32) / 32768.0) ** 2))
            silent_count = silent_count + 1 if rms < settings.silence_threshold else 0
            if silent_count >= silent_blocks_needed and len(frames) > silent_blocks_needed + 2:
                break
            if time.time() - start > settings.max_record_sec:
                log.warning("Max record duration reached.")
                break

    audio = np.concatenate(frames, axis=0)
    path = os.path.join(RECORDINGS_DIR, f"cmd_{int(time.time())}.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    log.info(f"Recorded {len(audio)/sr:.2f}s → {path}")
    return path

# ════════════════════════════════════════════════════════════════════
# 6. LOCAL TRANSCRIPTION (faster-whisper)
# ════════════════════════════════════════════════════════════════════

class LocalTranscriber:
    _model: WhisperModel | None = None

    def __init__(self):
        if LocalTranscriber._model is None:
            status(f"Loading Whisper '{settings.whisper_size}' on {settings.whisper_device}…")
            LocalTranscriber._model = WhisperModel(
                settings.whisper_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute,
                download_root="models",
            )
            log.info("Whisper model ready.")
        self.model = LocalTranscriber._model

    def transcribe(self, wav_path: str) -> str:
        segments, _ = self.model.transcribe(
            wav_path,
            language=settings.whisper_language or None,
            vad_filter=True,
            beam_size=1,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        log.info(f"Transcription: {text!r}")
        return text

# ════════════════════════════════════════════════════════════════════
# 7. OPENROUTER CLIENT (single key, retries)
# ════════════════════════════════════════════════════════════════════

class OpenRouterError(Exception):
    pass


class OpenRouterClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": settings.http_referer,
            "X-Title": settings.app_title,
            "Content-Type": "application/json",
        })

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, OpenRouterError)),
    )
    def chat(self, model: str, messages: list[dict], temperature: float = 0.4,
             max_tokens: int = 1024) -> str:
        url = f"{settings.openrouter_base_url}/chat/completions"
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        try:
            resp = self.session.post(url, json=payload, timeout=60)
        except requests.RequestException as e:
            log.error(f"Network error: {e}")
            raise
        if resp.status_code >= 400:
            log.error(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
            raise OpenRouterError(f"{resp.status_code} {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"].strip()

# ════════════════════════════════════════════════════════════════════
# 8. AGENTS (Planner → Critic → Executor)
# ════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    name: str = "Agent"
    model: str = ""
    system_prompt: str = ""

    def __init__(self, client: OpenRouterClient):
        self.client = client

    def _run(self, user_msg: str, temperature: float = 0.4) -> str:
        return self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
        )

    @abstractmethod
    def act(self, *args, **kwargs) -> str: ...


class PlannerAgent(BaseAgent):
    name = "Planner (OpenAI)"

    def __init__(self, client):
        super().__init__(client)
        self.model = settings.planner_model
        self.system_prompt = (
            "You are a concise planning agent. Given a user voice command, "
            "produce a clear, structured preliminary plan or draft answer in 4-8 short bullet points. "
            "Do NOT reveal private chain-of-thought; output only the plan. Be practical and actionable."
        )

    def act(self, user_command: str) -> str:
        return self._run(f"User command:\n{user_command}\n\nProduce the preliminary plan.")


class CriticAgent(BaseAgent):
    name = "Critic (Gemini)"

    def __init__(self, client):
        super().__init__(client)
        self.model = settings.critic_model
        self.system_prompt = (
            "You are a critical reviewer. You receive a preliminary plan from another agent. "
            "Identify weaknesses, missing steps, factual risks, and suggest concrete improvements. "
            "Output a REVISED, IMPROVED plan — clear and concise. "
            "Do not reveal hidden reasoning; provide only the improved plan plus a one-line note of key fixes."
        )

    def act(self, original_command: str, plan: str) -> str:
        return self._run(
            f"Original user command:\n{original_command}\n\n"
            f"Preliminary plan to critique:\n{plan}\n\n"
            "Return: (1) Improved plan, (2) one-line 'Key fixes:' summary.",
            temperature=0.3,
        )


class ExecutorAgent(BaseAgent):
    name = "Executor (Claude)"

    def __init__(self, client):
        super().__init__(client)
        self.model = settings.executor_model
        self.system_prompt = (
            "You are the lead executive agent. You receive an improved plan and the original user request. "
            "Produce the FINAL, user-facing answer. Be helpful, accurate, and concise. "
            "Speak directly to the user. Do not mention other agents or internal processes. "
            "Do not reveal chain-of-thought."
        )

    def act(self, original_command: str, improved_plan: str) -> str:
        return self._run(
            f"User asked:\n{original_command}\n\n"
            f"Vetted plan:\n{improved_plan}\n\n"
            "Now write the final answer to the user.",
            temperature=0.5,
        )

# ════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATOR (custom sequential pipeline)
# ════════════════════════════════════════════════════════════════════

def _summarize(text: str, max_len: int = 220) -> str:
    one = " ".join(text.split())
    return one if len(one) <= max_len else one[: max_len - 1] + "…"


class Orchestrator:
    def __init__(self):
        client = OpenRouterClient()
        self.planner = PlannerAgent(client)
        self.critic = CriticAgent(client)
        self.executor = ExecutorAgent(client)

    def run(self, user_command: str) -> str:
        agent_header(self.planner.name, self.planner.model)
        plan = self.planner.act(user_command)
        agent_summary(_summarize(plan))
        log.info(f"[Planner] {plan}")

        agent_header(self.critic.name, self.critic.model)
        improved = self.critic.act(user_command, plan)
        agent_summary(_summarize(improved))
        log.info(f"[Critic] {improved}")

        agent_header(self.executor.name, self.executor.model)
        answer = self.executor.act(user_command, improved)
        log.info(f"[Executor] {answer}")

        final_answer(answer)
        return answer

# ════════════════════════════════════════════════════════════════════
# 10. SESSION MEMORY
# ════════════════════════════════════════════════════════════════════

class SessionMemory:
    def __init__(self):
        self.turns: list[dict] = []
        if settings.enable_history:
            os.makedirs(os.path.dirname(settings.history_file), exist_ok=True)

    def add(self, user_text_in: str, final_answer_text: str) -> None:
        turn = {"ts": datetime.utcnow().isoformat(),
                "user": user_text_in, "answer": final_answer_text}
        self.turns.append(turn)
        if settings.enable_history:
            with open(settings.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def recent(self, n: int = 5) -> list[dict]:
        return self.turns[-n:]

# ════════════════════════════════════════════════════════════════════
# 11. MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def list_microphones() -> None:
    print("\nAvailable input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}  (sr={int(dev['default_samplerate'])})")
    print("Set MIC_DEVICE_INDEX in .env to choose one.\n")


def graceful_exit(signum, frame):
    print()
    status("Ctrl+C received — shutting down JARVIS. Goodbye.")
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, graceful_exit)
    banner("JARVIS-Local — Voice Assistant (OpenRouter Multi-Agent)")
    print(f"  Planner : {settings.planner_model}")
    print(f"  Critic  : {settings.critic_model}")
    print(f"  Executor: {settings.executor_model}")
    print(f"  Whisper : {settings.whisper_size} ({settings.whisper_device}/{settings.whisper_compute})")

    if "--list-mics" in sys.argv:
        list_microphones()
        return

    listener = WakeWordListener()
    transcriber = LocalTranscriber()
    orchestrator = Orchestrator()
    memory = SessionMemory()

    while True:
        try:
            listener.listen()
            wav_path = record_command()
            text = transcriber.transcribe(wav_path)

            if not text or len(text) < 2:
                err("No speech detected — back to listening.")
                continue

            user_text(text)
            answer = orchestrator.run(text)
            memory.add(text, answer)

        except KeyboardInterrupt:
            graceful_exit(None, None)
        except Exception as e:
            log.exception("Unhandled error in main loop")
            err(f"Error: {e} — returning to listening.")


if __name__ == "__main__":
    main()
