"""v6 grammar extension: core-jq constructs the v5 grammar never emitted — reduce,
foreach, walk, recursive descent, paths, if/then/elif classification, object-merge add,
computed-key group_by, INDEX. These are everyday jq (cumulative totals, deep transforms,
banding values, merging fragments) that were absent by historical accident; adding them
is generic coverage, not benchmark fitting: all NL phrasings are authored fresh in this
file and every emitted example is execution-verified by generate.py as usual.

try/catch is deliberately NOT generated: the schema system produces type-homogeneous
values, so catch branches would never fire and the examples would teach a dead construct.
"""

from .grammar_ext import _fields1, _pick


def _num1(ctx):
    """A top-level numeric field: (name, node) or None."""
    c = _fields1(ctx, ("int", "float"))
    return ctx.rng.choice(c) if c else None


def _str1(ctx):
    c = _fields1(ctx, ("str",))
    return ctx.rng.choice(c) if c else None


def g5_reduce_fold(ctx):
    """reduce .[].f as $x (init; op) — accumulate a single value step by step."""
    f = _num1(ctx)
    if not f:
        return None
    k, _ = f
    h, P = k.replace("_", " "), ctx.prefix
    roll = ctx.rng.random()
    if roll < 0.55:
        return {"program": f"{P}reduce .[].{k} as $v (0; . + $v)", "tier": 5,
                "tags": ["reduce", "sum"],
                "nl": [f"add up the {h} values one at a time into a single tally",
                       f"fold the {h}s into one running total and give me the end result",
                       f"accumulate all the {h} values into a grand total"]}
    if roll < 0.8:
        return {"program": f"{P}reduce .[].{k} as $v (0; if $v > . then $v else . end)",
                "tier": 5, "tags": ["reduce", "if", "max"],
                "nl": [f"scan through and keep the largest {h} seen so far — final answer only",
                       f"walk the list once, tracking the biggest {h}, and report it"]}
    return {"program": f"{P}reduce .[] as $r (0; . + 1)", "tier": 5,
            "tags": ["reduce", "count"],
            "nl": ["count the entries by folding over them one by one",
                   "tally how many records there are using a running counter"]}


def g5_foreach_running(ctx):
    """[foreach .[].f as $x (0; .+$x; .)] — the running/cumulative SERIES."""
    f = _num1(ctx)
    if not f:
        return None
    k, _ = f
    h, P = k.replace("_", " "), ctx.prefix
    return {"program": f"{P}[foreach .[].{k} as $v (0; . + $v; .)]", "tier": 5,
            "tags": ["foreach", "cumulative"],
            "nl": [f"the running total of {h} after each entry, as a list",
                   f"cumulative {h} step by step — one number per record",
                   f"show how the {h} total builds up entry by entry"]}


def g5_walk_numbers(ctx):
    """walk(if type == \"number\" then f(.) else . end) — deep numeric transform."""
    rng = ctx.rng
    roll = rng.random()
    if roll < 0.5:
        fac = rng.choice([2, 3, 10, 100])
        prog = f'walk(if type == "number" then . * {fac} else . end)'
        nl = [f"multiply every number anywhere in this structure by {fac}",
              f"scale each numeric value in here, however deep, by {fac}"]
        tags = ["walk", "arith"]
    else:
        prog = 'walk(if type == "number" then floor else . end)'
        nl = ["round every number in this document down to a whole number, wherever it is",
              "floor all numeric values throughout the structure"]
        tags = ["walk", "floor"]
    return {"program": prog, "tier": 5, "tags": tags, "nl": nl}


def g5_recurse_collect(ctx):
    """[.. | type-filter] — gather values at any depth."""
    roll = ctx.rng.random()
    if roll < 0.5:
        return {"program": "[.. | numbers] | add", "tier": 5,
                "tags": ["recurse", "sum"],
                "nl": ["sum every number in this document, no matter how deeply it's buried",
                       "the total of all numeric values anywhere in here"]}
    return {"program": "[.. | strings]", "tier": 5, "tags": ["recurse", "collect"],
            "order_insensitive": False,
            "nl": ["every string value anywhere in this structure",
                   "pull out all the strings, at any depth"]}


def g5_paths(ctx):
    """[leaf_paths] / [paths] — structural introspection."""
    if ctx.rng.random() < 0.6:
        return {"program": "[leaf_paths]", "tier": 5, "tags": ["paths"],
                "nl": ["the full path to every leaf value in this document",
                       "list each leaf's path as an array of keys/indices"]}
    return {"program": "[paths] | length", "tier": 5, "tags": ["paths", "count"],
            "nl": ["how many distinct paths exist in this structure",
                   "count every path through this document"]}


_BAND_WORDS = [("scant", "modest", "hefty"), ("dim", "fair", "bright"),
               ("shallow", "middling", "steep"), ("brief", "standard", "extended")]


def g5_if_band(ctx):
    """map(if .f >= hi then A elif .f >= lo then B else C end) — banding/classification."""
    f = _num1(ctx)
    if not f:
        return None
    k, node = f
    lo_v, hi_v = node.get("lo", 0), node.get("hi", 100)
    span = max(hi_v - lo_v, 3)
    t1 = round(lo_v + span / 3)
    t2 = round(lo_v + 2 * span / 3)
    if t1 >= t2:
        return None
    a, b, c = ctx.rng.choice(_BAND_WORDS)
    h, P = k.replace("_", " "), ctx.prefix
    return {"program": (f'{P}map(if .{k} >= {t2} then "{c}" '
                        f'elif .{k} >= {t1} then "{b}" else "{a}" end)'),
            "tier": 5, "tags": ["if", "classify"],
            "nl": [f'label each one "{c}" when {h} is at least {t2}, "{b}" from {t1}, '
                   f'otherwise "{a}"',
                   f'band the {h} values: {t2}+ is "{c}", {t1}-{t2} is "{b}", below that "{a}"']}


def g5_merge_add(ctx):
    """`add` over an array of objects = right-biased merge."""
    if not _fields1(ctx):
        return None
    return {"program": f"{ctx.prefix}add", "tier": 5, "tags": ["merge"],
            "nl": ["merge all of these into a single object — later entries win on conflicts",
                   "combine the records into one object, overwriting duplicate keys left to right",
                   "squash this list of objects down to one merged object"]}


def g5_group_computed(ctx):
    """group_by over a computed key (string slice)."""
    f = _str1(ctx)
    if not f:
        return None
    k, node = f
    if node.get("kind") not in ("name", "word", "city", "id"):
        return None
    h, P = k.replace("_", " "), ctx.prefix
    return {"program": f"{P}group_by(.{k}[0:1])", "tier": 5,
            "tags": ["group_by", "computed-key"], "order_insensitive": False,
            "nl": [f"group the records by the first letter of {h}",
                   f"bucket these by {h}'s initial character"]}


def g5_index_by(ctx):
    """INDEX(.f) — list of records -> object keyed by a field."""
    f = _str1(ctx)
    if not f:
        return None
    k, node = f
    if node.get("kind") not in ("id", "name", "word", "sku"):
        return None
    h, P = k.replace("_", " "), ctx.prefix
    return {"program": f"{P}INDEX(.{k})", "tier": 5, "tags": ["index"],
            "nl": [f"turn this list into an object keyed by {h}",
                   f"index the records by their {h} so I can look them up directly"]}


# weights tuned so T5 constructs land around ~5% of emitted rows (~100k in a 2M corpus)
RECORD_T5 = [
    (g5_reduce_fold, 1.8), (g5_foreach_running, 1.4), (g5_walk_numbers, 1.2),
    (g5_recurse_collect, 1.0), (g5_paths, 0.8), (g5_if_band, 1.8),
    (g5_merge_add, 1.0), (g5_group_computed, 1.0), (g5_index_by, 1.2),
]
