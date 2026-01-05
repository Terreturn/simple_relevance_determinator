# config.py
# Configuration dataclasses and env var loading


from dataclasses import dataclass
import os


def _env(name: str, default):
    """Read env var with automatic type casting from default."""
    val = os.getenv(name)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in {"1", "true", "yes"}
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


@dataclass
class LLMConfig:
    # ---- core ----
    model_name: str = _env("LLM_MODEL_NAME", "deepseek-chat")

    # ---- execution ----
    concurrency: int = _env("LLM_CONCURRENCY", 1)
    retry: int = _env("LLM_RETRY", 1)

    # ---- input control ----
    max_input_chars: int = _env("LLM_MAX_INPUT_CHARS", 12000)

    # ---- generation ----
    temperature: float = _env("LLM_TEMPERATURE", 0.0)
    timeout_s: float = _env("LLM_TIMEOUT_S", 60.0)


@dataclass
class AppConfig:
    llm: LLMConfig = LLMConfig()



CONFIG = AppConfig()



