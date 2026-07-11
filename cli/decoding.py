"""Input-grounded decoding: repair + execution-filter for generated jq programs.

The from-scratch model's dominant failure is emitting a training-vocabulary field name
instead of one present in the user's JSON (`map(.manager)` when the input has `sku`).
Grounding exploits what the CLI always has at hand — the input document:

  1. collect the set of field names actually present in the input (recursively);
  2. in a candidate program, rewrite any `.identifier` accessor whose name is NOT in the
     input to the best-matching real key (scored against both the hallucinated name and
     the words of the request);
  3. try candidates in order (greedy first, then samples), each raw and repaired, and
     return the first one that actually executes under jq.

This never adds knowledge the model lacks — it constrains its output to fields that
exist, exactly like a shell's tab-completion. Reported as a separate system row
("+ input-grounded decoding"), never silently folded into raw model scores.
"""
import difflib
import json
import re
import subprocess

# jq builtins/keywords that legitimately appear after a dot or as bare identifiers
_JQ_WORDS = {
    "length", "keys", "keys_unsorted", "values", "add", "unique", "unique_by", "sort",
    "sort_by", "group_by", "min_by", "max_by", "min", "max", "reverse", "flatten", "map",
    "map_values", "select", "has", "in", "any", "all", "not", "to_entries",
    "from_entries", "with_entries", "type", "test", "match", "capture", "split", "join",
    "ltrimstr", "rtrimstr", "startswith", "endswith", "ascii_downcase", "ascii_upcase",
    "tostring", "tonumber", "contains", "range", "floor", "ceil", "round", "fabs",
    "empty", "first", "last", "getpath", "paths", "leaf_paths", "recurse", "gsub", "sub",
    "then", "else", "elif", "end", "if", "and", "or", "reduce", "foreach", "as", "walk",
    "limit", "until", "while", "try", "catch", "input", "inputs", "explode", "implode",
    "ascii", "tojson", "fromjson", "INDEX", "GROUP_BY", "env", "now", "utf8bytelength",
}


def input_keys(doc):
    """All object keys present anywhere in the input document."""
    keys = set()
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            keys.update(node.keys())
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return keys


def _mask_strings(prog):
    """Replace "..." spans with placeholders so repairs never touch string literals."""
    lits = []

    def keep(m):
        lits.append(m.group(0))
        return f"\x00{len(lits) - 1}\x00"

    return re.sub(r'"(?:[^"\\]|\\.)*"', keep, prog), lits


def _unmask(prog, lits):
    return re.sub(r"\x00(\d+)\x00", lambda m: lits[int(m.group(1))], prog)


def _best_key(bad, keys, request):
    """Best replacement for a hallucinated field: similarity to the bad name, boosted
    when the key's words appear in the request."""
    req = request.lower()
    best, best_score = None, 0.0
    for k in keys:
        s = difflib.SequenceMatcher(None, bad.lower(), k.lower()).ratio()
        words = [w for w in k.lower().split("_") if len(w) > 2]
        if words and all(w in req for w in words):
            s += 0.6
        elif any(w in req for w in words):
            s += 0.25
        if s > best_score:
            best, best_score = k, s
    return best if best_score >= 0.35 else None


def repair_fields(program, keys, request):
    """Rewrite `.identifier` accessors not present in the input to best-match keys."""
    masked, lits = _mask_strings(program)

    def fix(m):
        name = m.group(1)
        if name in keys or name in _JQ_WORDS:
            return m.group(0)
        repl = _best_key(name, keys, request)
        return f".{repl}" if repl else m.group(0)

    fixed = re.sub(r"\.([A-Za-z_][A-Za-z0-9_]*)", fix, masked)
    return _unmask(fixed, lits)


def _informative(stdout):
    """An output stream that is empty or ALL nulls didn't answer anything — in jq,
    accessing a missing field succeeds and yields null, so hallucinated-field programs
    'run fine'. Such outputs must not satisfy the filter."""
    lines = [l for l in stdout.strip().splitlines() if l.strip()]
    if not lines:
        return False
    return not all(l.strip() == "null" for l in lines)


def grounded_pick(candidates, doc, request, jq_bin, timeout=1.0):
    """Try candidates in order — REPAIRED variant first (repairing is the point; a raw
    hallucinated-field program executes 'successfully' with null output in jq, so raw-
    first would mask every repair). First variant with an informative execution wins.
    Returns (program, meta) — meta records what happened, for diagnostics."""
    keys = input_keys(doc)
    doc_json = json.dumps(doc)
    seen = set()

    def runs(v):
        try:
            proc = subprocess.run([jq_bin, "-c", v], input=doc_json,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0 and _informative(proc.stdout)

    for cand in candidates:
        repaired = repair_fields(cand, keys, request)
        order = ((repaired, "repaired"), (cand, "raw")) if repaired != cand \
            else ((cand, "raw"),)
        for variant, kind in order:
            v = variant.strip()
            if not v or v in seen:
                continue
            seen.add(v)
            if runs(v):
                return v, {"picked": kind, "tried": len(seen)}
    fallback = candidates[0] if candidates else ""
    return fallback, {"picked": "fallback", "tried": len(seen)}
