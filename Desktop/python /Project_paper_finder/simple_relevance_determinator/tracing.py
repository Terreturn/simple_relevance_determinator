# tracing.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

logger = logging.getLogger("paper_finder")

# ---------- Logging and tracing utilities ----------
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

# ---------- JSONL event writer and timer ----------
def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


class JsonlEventWriter:
    """Append one JSON object per line for easy grep / pandas read_json(lines=True)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, event: dict[str, Any]) -> None:
        event = _jsonable(event)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class SpanTimer:
    """Simple timer context for latency_ms measurement."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)


# ---------- Error classification ----------
def classify_error(e: Exception) -> str:
    msg = (str(e) or "").lower()

    # very light heuristics; you can expand as you observe real errors
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if "token" in msg and ("limit" in msg or "maximum" in msg or "context" in msg):
        return "token_limit"
    if "json" in msg and ("decode" in msg or "valid json" in msg):
        return "json_parse"
    if "validation" in msg or "pydantic" in msg:
        return "schema_validation"
    return "unknown"
