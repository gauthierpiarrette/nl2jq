"""Novelty gate for the frozen benchmark (FROZEN_BENCH_DESIGN.md §c).

Rebuilds the training field vocabulary V_train from pipeline/schemas.py at check time
(so the gate stays correct if the training vocab later grows) and fails any bench item
whose input reuses it. Three rules:

  1. HARD DISJOINTNESS  — no input field name (recursive) may equal a V_train name.
  2. NEAR-MISS BAR      — no abbreviations/inflections of V_train names (prefix rule,
                          small edit distance, curated blocklist). Kills temp<-temperature,
                          msg<-message, seat<-seats.
  3. ENUM-VALUE GATE    — no input string value or program string literal may reuse a
                          training enum value ("paid", "error", "admin", "USD", ...).

Abstract single-letter keys (a,b,c,x,n,s,t) are allowed only on items tagged
abstract:true, quota-limited by the caller (<=5% of the set).

    python -m bench.validate_novelty items.jsonl        # gate a candidate file
    python -m bench.validate_novelty --dump vtrain.json # dump V_train for authoring
"""
import argparse
import json
import re
import sys
from pathlib import Path

ABSTRACT_KEYS = {"a", "b", "c", "x", "n", "s", "t", "k", "v"}
# Common abbreviations of vocabulary words that the edit/prefix rules can miss.
BLOCKLIST = {"msg", "tmp", "usr", "amt", "addr", "desc", "cat", "num", "val", "stat",
             "lvl", "cfg", "img", "pwd", "tel", "qty", "temp", "user", "dark", "ok"}


def build_vtrain():
    """Union every field-name pool in pipeline/schemas.py, incl. constructible compounds."""
    from pipeline import schemas as S

    names, enum_values = set(), set()

    def walk(node):
        if not isinstance(node, dict):
            return
        t = node.get("t")
        if t == "obj":
            for k, v in node.get("fields", {}).items():
                names.add(k)
                walk(v)
        elif t == "arr":
            walk(node.get("item"))
        elif t == "enum":
            enum_values.update(v for v in node.get("vals", []) if isinstance(v, str))

    for _noun, pool in S.DOMAINS.values():
        for name, leaf, _p in pool:
            names.add(name)
            walk(leaf)
    for name, leaf in S._COLLECTION_FIELDS:
        names.add(name)
        walk(leaf)
    for name, leaf in S._TEXT_FIELDS:
        names.add(name)
        walk(leaf)
    names.update(S._FLOAT_BASES)
    names.update(S._INT_BASES)
    for lst in S._STR_KINDS.values():
        names.update(lst)
    names.update(S._BOOL_NAMES)
    for k, vals in S._ENUM_FIELDS.items():
        names.add(k)
        enum_values.update(v for v in vals if isinstance(v, str))
    names.update(S._SETTINGS_KEYS)
    names.update(S._FLAG_KEYS)
    names.update({"key", "value"})  # kvpairs primitive
    # nested-array field names injected by sample_schema
    names.update({"orders", "members", "events", "builds", "transactions",
                  "readings", "repos"})
    compounds = ({f"{m}_{b}" for m in S._FLOAT_MODS for b in S._FLOAT_BASES}
                 | {f"{m}_{b}" for m in S._INT_MODS for b in S._INT_BASES})
    return names | compounds, enum_values


def _edit_distance(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def field_violation(field, vtrain):
    """Return a reason string if `field` collides with V_train, else None."""
    f = field.lower()
    if f in vtrain:
        return f"exact: {field}"
    if f in BLOCKLIST:
        return f"blocklist: {field}"
    # prefix rule: the whole shorter word (>=4 chars) is a prefix of the longer one
    for v in vtrain:
        short, long_ = (f, v) if len(f) <= len(v) else (v, f)
        if len(short) >= 4 and long_.startswith(short):
            return f"prefix-of-vocab: {field} ~ {v}"
    # small-edit rule
    for v in vtrain:
        lim = 1 if len(f) <= 5 else 2
        if _edit_distance(f, v, cap=lim) <= lim:
            return f"edit<={lim}: {field} ~ {v}"
    return None


def _input_fields(node, acc):
    if isinstance(node, dict):
        for k, v in node.items():
            acc.add(k)
            _input_fields(v, acc)
    elif isinstance(node, list):
        for v in node:
            _input_fields(v, acc)


def _input_strings(node, acc):
    if isinstance(node, dict):
        for v in node.values():
            _input_strings(v, acc)
    elif isinstance(node, list):
        for v in node:
            _input_strings(v, acc)
    elif isinstance(node, str):
        acc.add(node)


def check_item(item, vtrain, enum_values):
    """Return a list of violation strings for one bench item."""
    out = []
    fields = set()
    _input_fields(item["input"], fields)
    abstract = bool(item.get("abstract"))
    for f in sorted(fields):
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", f):
            out.append(f"non-identifier field: {f!r}")
            continue
        if f.lower() in ABSTRACT_KEYS:
            if not abstract:
                out.append(f"abstract key {f!r} on non-abstract item")
            continue
        v = field_violation(f, vtrain)
        if v:
            out.append(v)
    strings = set()
    _input_strings(item["input"], strings)
    for s in sorted(strings & enum_values):
        out.append(f"input value reuses training enum: {s!r}")
    for lit in re.findall(r'"([^"\\]*)"', item["reference_program"]):
        if lit in enum_values:
            out.append(f"program literal reuses training enum: {lit!r}")
    return out


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="?", help="candidate items .jsonl to gate")
    ap.add_argument("--dump", help="write V_train (names + enum values) to this JSON path")
    a = ap.parse_args()
    vtrain, enum_values = build_vtrain()
    if a.dump:
        Path(a.dump).write_text(json.dumps(
            {"names": sorted(vtrain), "enum_values": sorted(enum_values)}, indent=1))
        print(f"V_train: {len(vtrain)} names, {len(enum_values)} enum values -> {a.dump}")
        return
    items = [json.loads(l) for l in open(a.items)]
    bad = 0
    for it in items:
        v = check_item(it, vtrain, enum_values)
        if v:
            bad += 1
            print(f"{it.get('id', '?')}: " + "; ".join(v))
    n_abs = sum(bool(it.get("abstract")) for it in items)
    if n_abs > 0.05 * len(items):
        bad += 1
        print(f"abstract quota exceeded: {n_abs}/{len(items)} > 5%")
    print(f"\n{len(items) - bad}/{len(items)} items pass the novelty gate")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
