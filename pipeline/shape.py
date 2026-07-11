"""JSON context builders: RAW_PREFIX (truncated literal) and SHAPE (schema sketch)."""
import json

RAW_BUDGET = 1024
SHAPE_BUDGET = 900


def raw_prefix(doc) -> str:
    s = json.dumps(doc, separators=(", ", ": "))
    if len(s) <= RAW_BUDGET:
        return s
    if isinstance(doc, list):
        parts, size = [], 2
        for el in doc:
            es = json.dumps(el, separators=(", ", ": "))
            if size + len(es) + 2 > RAW_BUDGET:
                break
            parts.append(es)
            size += len(es) + 2
        if parts:
            return "[" + ", ".join(parts) + ", …]"
    return s[:RAW_BUDGET] + "…"


def _sketch(value, depth=0):
    if isinstance(value, dict):
        return {k: _sketch(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        n = len(value)
        inner = _sketch(value[0], depth + 1) if value else "empty"
        return [inner, f"×{n}"]
    if isinstance(value, str):
        ex = value if len(value) <= 24 else value[:24] + "…"
        return f"string ex:{json.dumps(ex)}"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return f"number ex:{value}"
    return "null"


def shape_sketch(doc) -> str:
    s = json.dumps(_sketch(doc), separators=(", ", ": "))
    if len(s) > SHAPE_BUDGET:
        s = s[:SHAPE_BUDGET] + "…"
    return s
