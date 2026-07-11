"""Shared paths, constants, and helpers for the nl2jq pipeline."""
import hashlib
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JQ_BIN = os.environ.get("JQ_BIN") or str(ROOT / "bin" / "jq")
DATA_DIR = ROOT / "data"

SPECIAL_TOKENS = ["<|pad|>", "<|request|>", "<|input|>", "<|program|>", "<|end|>"]


def format_example(request: str, context: str, program: str | None = None) -> str:
    """Canonical serialization for training and inference prompts."""
    s = f"<|request|> {request}\n<|input|> {context}\n<|program|>"
    if program is not None:
        s += f" {program}<|end|>"
    return s


def canon(value) -> str:
    """Canonical JSON string for output comparison (key-order-insensitive)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canon_outputs(outputs: list) -> str:
    """jq emits a stream of values; canonicalize the whole stream."""
    return "\n".join(canon(v) for v in outputs)


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]
