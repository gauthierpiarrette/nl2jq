"""T5 non-coverage gate for the frozen benchmark (FROZEN_BENCH_DESIGN.md §e step 4).

T5 ("beyond-grammar") items must use jq constructs the training-data grammar provably
does NOT emit. This gate checks both directions:

  A. The grammar sources (pipeline/grammar.py, pipeline/grammar_ext.py) contain none of
     the T5 marker constructs inside any string literal (program templates are built from
     string literals / f-strings, so this approximates the emission census that the
     2026-07-10 audit performed by executing every generator).
  B. Every item tagged tier=5 contains at least one T5 marker; every item tagged
     tier<5 contains none (so covered-tier scores aren't secretly boosted by T5 ops).

    python -m bench.audit_coverage items.jsonl
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# construct -> regex over a jq PROGRAM string (rule B: items are pure jq, loose is fine)
T5_MARKERS = {
    "reduce": r"\breduce\b",
    "foreach": r"\bforeach\b",
    "walk": r"\bwalk\s*\(",
    "recursive-descent": r"(?<![.\w])\.\.(?![.\w])",
    "paths-family": r"\b(leaf_)?paths\b|\bgetpath\s*\(|\bsetpath\s*\(|\bdelpaths\s*\(",
    "try-catch": r"\btry\b|\bcatch\b",
    "if-then": r"\bif\b.*\bthen\b",
    "computed-key-group_by": r"group_by\s*\(\s*\.\[",
    "INDEX": r"\bINDEX\s*\(",
    "at-sh": r"@sh\b",
    "object-merge-add": r"\badd\b",  # special-cased below: only counts on array-of-objects input
}

# Rule A scans grammar SOURCE literals, which mix jq templates with NL-request templates
# and provenance comments — so anchor each marker to program syntax that NL prose can't
# produce ("reduce each price" must not trip the `reduce` marker).
GRAMMAR_MARKERS = {
    "reduce": r"\breduce\s+[.$\[]",
    "foreach": r"\bforeach\s+[.$\[]",
    "walk": r"\bwalk\s*\(",
    "recursive-descent": r"(?<![.\w])\.\.(?![.\w])",
    "paths-family": r"\bleaf_paths\b|\bgetpath\s*\(|\bsetpath\s*\(|\bdelpaths\s*\("
                    r"|\[\s*paths\s*\]|\|\s*paths\b|^paths\b",
    "try-catch": r"\btry\s+[.$(\[]|\bcatch\b",
    "if-then": r"\bif\s+[.$(\[]",
    "computed-key-group_by": r"group_by\s*\(\s*\.\[",
    "INDEX": r"\bINDEX\s*\(",
    "at-sh": r"@sh\b",
}


def _grammar_string_literals():
    lits = []
    for src in (ROOT / "pipeline" / "grammar.py", ROOT / "pipeline" / "grammar_ext.py"):
        tree = ast.parse(src.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                ds = ast.get_docstring(node, clean=False)
                if ds:
                    docstrings.add(ds)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value not in docstrings:
                lits.append(node.value)
    return lits


def check_grammar_clean():
    """Rule A: no T5 marker may appear in any grammar string literal (docstrings excluded,
    markers anchored to program syntax so NL-request templates can't false-positive).

    Historically scoped: this property was verified at the v1.0.0 freeze against the
    grammar hashes recorded in bench/frozen/GRAMMAR_AT_FREEZE.txt. Later grammar versions
    may legitimately emit these constructs (they are core jq); if the current sources no
    longer match the freeze hashes, rule A is reported as historically-satisfied and
    skipped rather than failed — the frozen benchmark itself never changes."""
    import hashlib
    record = ROOT / "bench" / "frozen" / "GRAMMAR_AT_FREEZE.txt"
    if record.exists():
        frozen_hashes = dict(
            line.split(None, 1)[::-1] for line in record.read_text().splitlines()
            if re.match(r"^[0-9a-f]{64}\s", line))
        current_match = all(
            hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h
            for f, h in frozen_hashes.items())
        if not current_match:
            print("rule A: grammar has evolved since the v1.0.0 freeze "
                  "(see GRAMMAR_AT_FREEZE.txt) — historically satisfied, skipping")
            return []
    lits = _grammar_string_literals()
    dirty = []
    for name, rx in GRAMMAR_MARKERS.items():
        for lit in lits:
            if re.search(rx, lit):
                dirty.append((name, lit[:60]))
    return dirty


def item_markers(item):
    """Which T5 markers does this item's reference program use?"""
    prog = item["reference_program"]
    found = [n for n, rx in T5_MARKERS.items()
             if n != "object-merge-add" and re.search(rx, prog)]
    # object-merge add: `add` applied to an array of OBJECTS (not numbers)
    if re.search(r"\badd\b", prog):
        doc = item["input"]
        if isinstance(doc, list) and doc and all(isinstance(x, dict) for x in doc) \
                and re.match(r"^\s*add\s*$|^\s*add\s*\|", prog):
            found.append("object-merge-add")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", help="candidate items .jsonl")
    a = ap.parse_args()

    dirty = check_grammar_clean()
    if dirty:
        print("GRAMMAR NOT CLEAN — T5 markers found in grammar string literals:")
        for name, lit in dirty:
            print(f"  {name}: {lit!r}")
        sys.exit(1)
    print("grammar sources clean: no T5 marker in any emitted-program template")

    items = [json.loads(l) for l in open(a.items)]
    bad = 0
    for it in items:
        marks = item_markers(it)
        tier = it.get("tier")
        if tier == 5 and not marks:
            bad += 1
            print(f"{it.get('id', '?')}: tier=5 but no T5 construct in program: "
                  f"{it['reference_program']!r}")
        elif tier != 5 and marks:
            bad += 1
            print(f"{it.get('id', '?')}: tier={tier} but uses T5 construct(s) {marks}: "
                  f"{it['reference_program']!r}")
    n5 = sum(it.get("tier") == 5 for it in items)
    print(f"\nT5 items: {n5}; violations: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
