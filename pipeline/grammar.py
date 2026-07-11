"""Type-directed jq program sampling with paired NL description templates (M0: tier 1 + light tier 2).

Each generator inspects the schema (so paths/types are valid by construction) and the
sampled documents (so filter constants are realistic), and returns:
    {"program": str, "nl": [variants...], "tier": int, "tags": [...]}
"""
import json
import random


def humanize(name: str) -> str:
    return name.replace("_", " ")


def pluralize(noun: str) -> str:
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    if " " in noun:  # e.g. "log entry" -> "log entries"
        head, _, tail = noun.rpartition(" ")
        return head + " " + pluralize(tail)
    return noun + "s"


def path_phrase(path: tuple) -> str:
    return " ".join(humanize(p) for p in path)


def jq_path(path: tuple) -> str:
    return "." + ".".join(path)


def leaf_paths(obj_node: dict, prefix=(), include_optional=True):
    """(path, leaf_node, is_optional) for every non-array leaf reachable through objects."""
    out = []
    for name, child in obj_node["fields"].items():
        opt = name in obj_node.get("optional", [])
        if not include_optional and opt:
            continue
        p = prefix + (name,)
        if child["t"] == "obj":
            out.extend([(pp, nn, oo or opt) for pp, nn, oo in
                        leaf_paths(child, p, include_optional)])
        elif child["t"] != "arr":
            out.append((p, child, opt))
    return out


def _by_type(paths, types, optional_ok=False):
    return [(p, n) for p, n, opt in paths if n["t"] in types and (optional_ok or not opt)]


def _values_at(records, path):
    vals = []
    for r in records:
        v = r
        for seg in path:
            v = v.get(seg) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None:
            vals.append(v)
    return vals


def _fmt_num(v):
    return str(int(v)) if float(v) == int(v) else f"{v:g}"


class Ctx:
    """Generation context for one array-of-records view of the SHOWN document.

    `records` are the records of docs[0] only, so filter/equality constants sampled
    from them keep the shown example non-degenerate.
    """

    def __init__(self, item_node, records, noun, rng, prefix=""):
        self.item = item_node
        self.records = records
        self.noun = noun          # e.g. "order"
        self.plural = pluralize(noun)
        self.rng = rng
        self.prefix = prefix      # e.g. ".orders | " when the array is nested in an object
        self.paths = leaf_paths(item_node)

    def pick(self, types, optional_ok=False):
        cands = _by_type(self.paths, types, optional_ok)
        return self.rng.choice(cands) if cands else None


# ---------------------------------------------------------------- conditions

def sample_condition(ctx: Ctx):
    """Returns (jq_cond, bare_nl_clauses) or None.

    NL clauses are BARE predicates ("price is over 168", "status is paid") with no
    leading connector — the calling generator owns "where"/"with"/"whose".
    Filter constants are drawn from the FIRST document so the shown example is
    guaranteed non-degenerate.
    """
    rng = ctx.rng
    doc0 = ctx.records  # ctx holds the shown document's records
    kind = rng.choices(["eq", "cmp", "bool"], weights=[4, 3, 2])[0]
    if kind == "bool":
        c = ctx.pick(("bool",))
        if not c:
            return None
        path, _ = c
        ph = path_phrase(path)
        if rng.random() < 0.5:
            return f"select({jq_path(path)})", [f"{ph} is true", f"{ph} is enabled"]
        return f"select({jq_path(path)} | not)", [f"{ph} is false", f"{ph} is disabled"]
    if kind == "eq":
        c = ctx.pick(("enum", "str", "int"))
        if not c:
            return None
        path, node = c
        vals = _values_at(doc0, path) or _values_at(ctx.records, path)
        if not vals:
            return None
        v = rng.choice(vals)
        ph = path_phrase(path)
        vj = json.dumps(v)
        vn = v if isinstance(v, str) else _fmt_num(v)
        op, opj = rng.choices([("is", "=="), ("is not", "!=")], weights=[6, 1])[0]
        clauses = [f"{ph} {op} {vn}",
                   f"{ph} {'equal to' if op == 'is' else 'different from'} {vn}"]
        if op == "is":
            clauses.append(f"{ph} {vn}")  # compact: "status paid", "level error"
        return (f"select({jq_path(path)} {opj} {vj})", clauses)
    c = ctx.pick(("int", "float"))
    if not c:
        return None
    path, node = c
    vals = sorted(_values_at(doc0, path))
    if len(vals) < 2:
        return None
    v = _fmt_num(vals[len(vals) // 2])  # median of doc0 -> non-degenerate on the shown doc
    ph = path_phrase(path)
    opj, words = ctx.rng.choice([(">", ["greater than", "over", "above", "more than"]),
                                 ("<", ["less than", "under", "below"]),
                                 (">=", ["at least"]), ("<=", ["at most"])])
    w = ctx.rng.choice(words)
    return (f"select({jq_path(path)} {opj} {v})",
            [f"{ph} is {w} {v}", f"{ph} {opj} {v}"])


# ---------------------------------------------------------------- generators
# Each takes Ctx, returns task dict or None. `P` is ctx.prefix.

def g_pluck(ctx):
    c = ctx.pick(("str", "enum", "int", "float", "bool"), optional_ok=False)
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    body = ctx.rng.choice([f"map({jq_path(path)})", f"[.[] | {jq_path(path)}]"])
    return {"program": f"{P}{body}", "tier": 1, "tags": ["pluck"],
            "nl": [f"get the {ph} of every {ctx.noun}", f"list all {ctx.plural} {ph}s",
                   f"extract {ph} from each {ctx.noun}", f"just the {ph}s",
                   f"all the {ph} values"]}


def g_filter(ctx):
    cond = sample_condition(ctx)
    if not cond:
        return None
    jc, nls = cond
    P, nl = ctx.prefix, ctx.rng.choice
    body = ctx.rng.choice([f"map({jc})", f"[.[] | {jc}]"])
    return {"program": f"{P}{body}", "tier": 1, "tags": ["filter"],
            "nl": [f"only the {ctx.plural} where {nl(nls)}", f"{ctx.plural} where {nl(nls)}",
                   f"show {ctx.plural} where {nl(nls)}", f"filter to {ctx.plural} where {nl(nls)}",
                   f"keep {ctx.plural} where {nl(nls)}", f"{ctx.plural} with {nl(nls)}",
                   f"the {ctx.plural} that have {nl(nls)}"]}


def g_filter_pluck(ctx):
    cond = sample_condition(ctx)
    tgt = ctx.pick(("str", "enum", "int", "float"))
    if not cond or not tgt:
        return None
    jc, nls = cond
    tpath, _ = tgt
    tph, P = path_phrase(tpath), ctx.prefix
    body = ctx.rng.choice([f"map({jc} | {jq_path(tpath)})",
                           f"[.[] | {jc} | {jq_path(tpath)}]",
                           f"map({jc}) | map({jq_path(tpath)})"])
    nl = ctx.rng.choice
    return {"program": f"{P}{body}", "tier": 1, "tags": ["filter", "pluck"],
            "nl": [f"the {tph} of {ctx.plural} where {nl(nls)}",
                   f"{tph}s for {ctx.plural} where {nl(nls)}",
                   f"get {tph} for every {ctx.noun} where {nl(nls)}"]}


def g_count(ctx):
    P = ctx.prefix
    return {"program": f"{P}length", "tier": 1, "tags": ["count"],
            "nl": [f"how many {ctx.plural} are there", f"count the {ctx.plural}",
                   f"number of {ctx.plural}", f"total count of {ctx.plural}"]}


def g_count_where(ctx):
    cond = sample_condition(ctx)
    if not cond:
        return None
    jc, nls = cond
    P, nl = ctx.prefix, ctx.rng.choice
    body = ctx.rng.choice([f"map({jc}) | length", f"[.[] | {jc}] | length"])
    return {"program": f"{P}{body}", "tier": 1, "tags": ["count", "filter"],
            "nl": [f"how many {ctx.plural} have {nl(nls)}", f"count of {ctx.plural} where {nl(nls)}",
                   f"number of {ctx.plural} where {nl(nls)}"]}


def g_index(ctx):
    P = ctx.prefix
    which, idx = ctx.rng.choice([("first", ".[0]"), ("last", ".[-1]")])
    if ctx.rng.random() < 0.5:
        return {"program": f"{P}{idx}", "tier": 1, "tags": ["index"],
                "nl": [f"the {which} {ctx.noun}", f"show the {which} entry",
                       f"{which} {ctx.noun} in the list"]}
    c = ctx.pick(("str", "enum", "int", "float"))
    if not c:
        return None
    path, _ = c
    ph = path_phrase(path)
    return {"program": f"{P}{idx}{jq_path(path)}", "tier": 1, "tags": ["index", "pluck"],
            "nl": [f"the {ph} of the {which} {ctx.noun}", f"{which} {ctx.noun}'s {ph}",
                   f"get {ph} from the {which} entry"]}


def g_slice(ctx):
    n = ctx.rng.randint(2, min(4, max(2, len(ctx.records) - 1)))
    P = ctx.prefix
    return {"program": f"{P}.[0:{n}]", "tier": 1, "tags": ["slice"],
            "nl": [f"the first {n} {ctx.plural}", f"just the top {n} entries",
                   f"first {n} {ctx.plural} only"]}


def g_sum(ctx):
    c = ctx.pick(("int", "float"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    body = ctx.rng.choice([f"map({jq_path(path)}) | add", f"[.[] | {jq_path(path)}] | add"])
    return {"program": f"{P}{body}", "tier": 2, "tags": ["sum"],
            "nl": [f"sum of all {ph}s", f"total {ph} across {ctx.plural}",
                   f"add up the {ph} of every {ctx.noun}", f"combined {ph}"]}


def g_unique(ctx):
    c = ctx.pick(("str", "enum", "int"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}[.[] | {jq_path(path)}] | unique", "tier": 2, "tags": ["unique"],
            "nl": [f"the distinct {ph} values", f"unique {ph}s", f"all different {ph}s, no duplicates"]}


def g_keys(ctx):
    P = ctx.prefix
    return {"program": f"{P}.[0] | keys", "tier": 1, "tags": ["keys"],
            "nl": [f"what fields does a {ctx.noun} have", f"list the keys of a {ctx.noun}",
                   f"field names in each {ctx.noun}"]}


def g_arith(ctx):
    nums = _by_type(ctx.paths, ("int", "float"), optional_ok=False)
    if len(nums) < 2:
        return None
    (p1, _), (p2, _) = ctx.rng.sample(nums, 2)
    ph1, ph2, P = path_phrase(p1), path_phrase(p2), ctx.prefix
    op, word = ctx.rng.choice([("*", "multiplied by"), ("+", "plus"), ("-", "minus")])
    return {"program": f"{P}map({jq_path(p1)} {op} {jq_path(p2)})", "tier": 1, "tags": ["arith"],
            "nl": [f"{ph1} {word} {ph2} for each {ctx.noun}",
                   f"compute {ph1} {op} {ph2} per {ctx.noun}"]}


def g_alt_default(ctx):
    opts = [(p, n) for p, n, o in ctx.paths if o and n["t"] in ("str", "enum")]
    reqs = _by_type(ctx.paths, ("str", "enum"), optional_ok=False)
    if not opts or not reqs:
        return None
    (po, _), (pr, _) = ctx.rng.choice(opts), ctx.rng.choice(reqs)
    pho, phr, P = path_phrase(po), path_phrase(pr), ctx.prefix
    return {"program": f"{P}map({jq_path(po)} // {jq_path(pr)})", "tier": 1, "tags": ["alternative"],
            "nl": [f"the {pho}, falling back to {phr} when missing",
                   f"{pho} or {phr} if there is none",
                   f"each {ctx.noun}'s {pho}, defaulting to {phr}"]}


def g_has(ctx):
    opts = [(p, n) for p, n, o in ctx.paths if o and len(p) == 1]
    if not opts:
        return None
    (path, _) = ctx.rng.choice(opts)
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}map(select(has({json.dumps(path[0])})))", "tier": 1, "tags": ["has"],
            "nl": [f"{ctx.plural} that have a {ph}", f"only entries with a {ph} field",
                   f"{ctx.plural} where {ph} is present"]}


def g_sort_by(ctx):
    c = ctx.pick(("int", "float", "str", "enum"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    if ctx.rng.random() < 0.5:
        return {"program": f"{P}sort_by({jq_path(path)})", "tier": 2, "tags": ["sort"],
                "nl": [f"the {ctx.plural} sorted by {ph}", f"sort the {ctx.plural} by {ph}",
                       f"{ctx.plural} in order of {ph}", f"order the {ctx.plural} by {ph}"]}
    tgt = ctx.pick(("str", "enum", "int", "float"))
    if not tgt:
        return None
    tp, _ = tgt
    tph = path_phrase(tp)
    return {"program": f"{P}sort_by({jq_path(path)}) | map({jq_path(tp)})",
            "tier": 2, "tags": ["sort", "pluck"],
            "nl": [f"the {tph}s of {ctx.plural} sorted by {ph}",
                   f"{tph} for each {ctx.noun}, ordered by {ph}",
                   f"list {tph} sorted by {ph}"]}


def g_min_max_by(ctx):
    c = ctx.pick(("int", "float"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    fn, words = ctx.rng.choice([("max_by", ["highest", "most", "largest", "greatest"]),
                                ("min_by", ["lowest", "least", "smallest"])])
    w = ctx.rng.choice(words)
    if ctx.rng.random() < 0.5:
        return {"program": f"{P}{fn}({jq_path(path)})", "tier": 2, "tags": [fn],
                "nl": [f"the {ctx.noun} with the {w} {ph}",
                       f"which {ctx.noun} has the {w} {ph}",
                       f"the {w}-{ph} {ctx.noun}"]}
    tgt = ctx.pick(("str", "enum"))
    if not tgt:
        return None
    tp, _ = tgt
    tph = path_phrase(tp)
    return {"program": f"{P}{fn}({jq_path(path)}) | {jq_path(tp)}", "tier": 2, "tags": [fn, "pluck"],
            "nl": [f"the {tph} of the {ctx.noun} with the {w} {ph}",
                   f"{tph} of the {w}-{ph} {ctx.noun}"]}


def g_group_count(ctx):
    c = ctx.pick(("enum", "str"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    key = path[-1]
    return {"program": f"{P}group_by({jq_path(path)}) | "
                       f"map({{{key}: .[0]{jq_path(path)}, count: length}})",
            "tier": 3, "tags": ["group_by", "count"],
            "nl": [f"the count of {ctx.plural} per {ph}",
                   f"how many {ctx.plural} for each {ph}",
                   f"number of {ctx.plural} grouped by {ph}",
                   f"{ctx.plural} count by {ph}"]}


def g_avg(ctx):
    c = ctx.pick(("int", "float"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}map({jq_path(path)}) | add / length", "tier": 2, "tags": ["arith", "avg"],
            "nl": [f"the average {ph}", f"mean {ph} across {ctx.plural}",
                   f"average {ph} of the {ctx.plural}", f"the mean {ph}"]}


ARRAY_GENERATORS = [(g_pluck, 3), (g_filter, 3), (g_filter_pluck, 2.5), (g_count, 1.2),
                    (g_count_where, 2), (g_index, 1.5), (g_slice, 1), (g_sum, 1.5),
                    (g_unique, 1), (g_keys, 0.8), (g_arith, 1), (g_alt_default, 0.8),
                    (g_has, 0.8), (g_sort_by, 1.5), (g_min_max_by, 1.5),
                    (g_group_count, 1.2), (g_avg, 1.2)]


# ---------------------------------------------------------- primitive generators
# Operate directly on the shown document (a list of numbers/strings, a nested list,
# or a bare string); the program runs on the top-level value with no path prefix.

def _weighted(choices, rng):
    tasks, weights = zip(*choices)
    t = rng.choices(tasks, weights=weights)[0]
    return {"program": t["program"], "tier": 2, "tags": t["tags"], "nl": t["nl"]}


def gp_num(doc0, rng):
    nums = [v for v in doc0 if isinstance(v, (int, float))]
    if len(nums) < 2:
        return None
    ch = [
        ({"program": "add", "tags": ["sum"],
          "nl": ["the sum of the numbers", "total of all the numbers", "add them all up",
                 "sum them", "the total"]}, 3),
        ({"program": "add / length", "tags": ["arith", "avg"],
          "nl": ["the average of the numbers", "the mean", "average value",
                 "the mean of the list"]}, 2),
        ({"program": "min", "tags": ["min"],
          "nl": ["the smallest number", "the minimum", "the lowest value", "min of the list"]}, 2),
        ({"program": "max", "tags": ["max"],
          "nl": ["the largest number", "the maximum", "the highest value", "max of the list"]}, 2),
        ({"program": "sort", "tags": ["sort"],
          "nl": ["sort the numbers", "the numbers in ascending order", "sorted ascending",
                 "sort them"]}, 2),
        ({"program": "sort | reverse", "tags": ["sort", "reverse"],
          "nl": ["sort the numbers descending", "largest to smallest",
                 "sorted in descending order"]}, 1.5),
        ({"program": "reverse", "tags": ["reverse"],
          "nl": ["the list reversed", "in reverse order", "reverse the numbers"]}, 1),
        ({"program": "unique", "tags": ["unique"],
          "nl": ["the distinct numbers", "unique values", "the numbers without duplicates",
                 "deduplicate them"]}, 1.5),
        ({"program": "length", "tags": ["count"],
          "nl": ["how many numbers are there", "the count of numbers", "how many values"]}, 1.5),
        ({"program": "map(. * .)", "tags": ["arith"],
          "nl": ["each number squared", "the squares of the numbers", "square every number"]}, 1),
    ]
    if all(float(v) == int(v) for v in nums):
        ch.append(({"program": "map(select(. % 2 == 0))", "tags": ["filter", "arith"],
                    "nl": ["only the even numbers", "the even values", "keep the evens"]}, 1))
        ch.append(({"program": "map(select(. % 2 == 1))", "tags": ["filter", "arith"],
                    "nl": ["only the odd numbers", "the odd values", "keep the odds"]}, 0.7))
    m = _fmt_num(sorted(nums)[len(nums) // 2])
    ch.append(({"program": f"map(select(. > {m}))", "tags": ["filter"],
                "nl": [f"the numbers over {m}", f"values greater than {m}", f"numbers above {m}"]}, 1))
    ch.append(({"program": f"map(select(. > {m})) | length", "tags": ["count", "filter"],
                "nl": [f"how many numbers are over {m}", f"count of values above {m}"]}, 1))
    n = min(3, len(nums) - 1)
    if n >= 2:
        ch.append(({"program": f"sort | reverse | .[0:{n}]", "tags": ["sort", "slice"],
                    "nl": [f"the {n} largest numbers", f"the top {n} values",
                           f"the {n} biggest numbers"]}, 1))
    return _weighted(ch, rng)


def gp_str(doc0, rng):
    strs = [v for v in doc0 if isinstance(v, str)]
    if len(strs) < 2:
        return None
    sep, sepword = rng.choice([(", ", "commas"), (",", "a comma"), (" ", "spaces"), (" - ", "dashes")])
    ch = [
        ({"program": f"join({json.dumps(sep)})", "tags": ["string", "join"],
          "nl": [f"join the values with {sepword}", f"combine them separated by {sepword}",
                 f"join into one string with {sepword}"]}, 2),
        ({"program": "map(ascii_upcase)", "tags": ["string"],
          "nl": ["uppercase each value", "all in capitals", "upper-case every string"]}, 1.5),
        ({"program": "map(ascii_downcase)", "tags": ["string"],
          "nl": ["lowercase each value", "all in lower case", "lower-case every string"]}, 1.5),
        ({"program": "map(length)", "tags": ["string"],
          "nl": ["the length of each value", "how many characters in each", "the lengths"]}, 1.5),
        ({"program": "sort", "tags": ["sort"],
          "nl": ["sort them alphabetically", "in alphabetical order", "sorted"]}, 1.5),
        ({"program": "unique", "tags": ["unique"],
          "nl": ["the distinct values", "without duplicates", "unique values", "deduplicate them"]}, 1.5),
        ({"program": "reverse", "tags": ["reverse"],
          "nl": ["in reverse order", "the list reversed"]}, 0.8),
        ({"program": "length", "tags": ["count"],
          "nl": ["how many values are there", "the number of items", "how many"]}, 1),
        ({"program": "sort_by(length)", "tags": ["sort", "string"],
          "nl": ["sort by length", "shortest to longest", "ordered by length"]}, 0.7),
    ]
    pre = rng.choice(strs)[0:1]
    if pre:
        ch.append(({"program": f"map(select(startswith({json.dumps(pre)})))", "tags": ["filter", "string"],
                    "nl": [f"the values starting with {pre}", f"those that start with {pre}"]}, 0.8))
    return _weighted(ch, rng)


def gp_nested(doc0, rng):
    if not (isinstance(doc0, list) and doc0 and all(isinstance(x, list) for x in doc0)):
        return None
    ch = [
        ({"program": "flatten", "tags": ["flatten"],
          "nl": ["flatten the nested lists into one array", "flatten them all",
                 "merge into a single flat list", "flatten everything"]}, 2),
        ({"program": "flatten(1)", "tags": ["flatten"],
          "nl": ["flatten one level only", "flatten a single level", "flatten by one level"]}, 1),
        ({"program": "map(length)", "tags": ["count"],
          "nl": ["the length of each sublist", "how many items in each group",
                 "the size of each list"]}, 1.5),
        ({"program": "map(add)", "tags": ["sum"],
          "nl": ["the sum of each sublist", "the total of each group", "sum each list"]}, 1),
        ({"program": "length", "tags": ["count"],
          "nl": ["how many groups are there", "the number of sublists"]}, 0.8),
    ]
    return _weighted(ch, rng)


def gp_scalar(doc0, rng):
    if not isinstance(doc0, str) or not doc0:
        return None
    ch = []
    if "/" in doc0:
        ch.append(({"program": 'split("/")', "tags": ["string", "split"],
                    "nl": ["split the path into segments", "split it on slashes",
                           "break the path into parts", "the path segments"]}, 2))
    if " " in doc0:
        ch.append(({"program": 'split(" ")', "tags": ["string", "split"],
                    "nl": ["split into words", "the words", "break it into words on spaces",
                           "split on spaces"]}, 2))
    ch.append(({"program": "length", "tags": ["string"],
                "nl": ["how many characters", "the length of the string", "the character count"]}, 1))
    ch.append(({"program": "ascii_upcase", "tags": ["string"],
                "nl": ["in uppercase", "upper-case it", "make it uppercase"]}, 1))
    ch.append(({"program": "ascii_downcase", "tags": ["string"],
                "nl": ["in lowercase", "lower-case it", "make it lowercase"]}, 1))
    return _weighted(ch, rng)


def sample_primitive_task(top, doc0, rng):
    if top["t"] == "arr" and top["item"]["t"] == "arr":
        return gp_nested(doc0, rng)
    if top["t"] == "arr" and top["item"]["t"] == "obj":      # key/value pairs -> from_entries
        return _ext.gp_kvpairs(doc0, rng)
    if top["t"] == "arr" and top["item"]["t"] in ("int", "float"):
        return _ext.prim_num_ext(doc0, rng) if rng.random() < 0.4 else gp_num(doc0, rng)
    if top["t"] == "arr" and top["item"]["t"] in ("str", "enum"):
        ns = _ext.gp_numstr(doc0, rng)                       # numeric strings -> tonumber ops
        if ns is not None and rng.random() < 0.8:
            return ns
        return _ext.prim_str_ext(doc0, rng) if rng.random() < 0.4 else gp_str(doc0, rng)
    if top["t"] == "str":
        return gp_scalar(doc0, rng)
    return None


def g_obj_path(schema_info, docs, rng):
    top = schema_info["schema"]
    paths = leaf_paths(top)
    cands = [(p, n) for p, n, o in paths if not o]
    if not cands:
        return None
    path, _ = rng.choice(cands)
    ph = path_phrase(path)
    return {"program": jq_path(path), "tier": 1, "tags": ["path"],
            "nl": [f"get the {ph}", f"what is the {ph}", f"show me the {ph}",
                   f"the {ph} value", f"pull out {ph}"]}


def g_obj_keys(schema_info, docs, rng):
    return {"program": "keys", "tier": 1, "tags": ["keys"],
            "nl": ["list the top-level keys", "what fields are in this object",
                   "show all field names"]}


def sample_task(schema_info: dict, docs: list, rng: random.Random):
    """Sample one task appropriate to the schema shape. Returns task dict or None.

    Constants are drawn from docs[0] (the document shown to the model) so the shown
    example is non-degenerate; the orchestrator independently re-executes on all docs.
    """
    top = schema_info["schema"]
    doc0 = docs[0]
    # primitive shapes (scalar/nested arrays, bare strings) get their own generators
    if schema_info.get("domain") == "primitive":
        return sample_primitive_task(top, doc0, rng)
    # homogeneous objects (settings = all numbers, flags = all booleans)
    if schema_info.get("domain") == "settings_num":
        return _ext.sample_settings_task(schema_info, docs, rng)
    if schema_info.get("domain") == "flags_bool":
        return _ext.sample_flags_task(schema_info, docs, rng)
    if top["t"] == "arr":
        ctx = Ctx(top["item"], doc0, schema_info["noun"], rng)
        gens, weights = zip(*ARRAY_GENERATORS)
        return rng.choices(gens, weights=weights)[0](ctx)
    # object top-level: sometimes operate on a nested array-of-records, otherwise
    # apply a whole-object op (path pluck / keys / to_entries / values / length).
    arr = top.get("_arr_field")
    if arr and rng.random() < 0.55:
        field, sub_noun = arr
        records = doc0.get(field, [])
        if len(records) >= 2:
            ctx = Ctx(top["fields"][field]["item"], records, sub_noun, rng, prefix=f".{field} | ")
            gens, weights = zip(*ARRAY_GENERATORS)
            return rng.choices(gens, weights=weights)[0](ctx)
    obj_gens = [(g_obj_path, 1.4), (g_obj_keys, 0.7),
                (_ext.go_to_entries, 1.0), (_ext.go_values, 1.0), (_ext.go_obj_length, 0.7)]
    gens, weights = zip(*obj_gens)
    return rng.choices(gens, weights=weights)[0](schema_info, docs, rng)


# v4 grammar extension. Imported last so grammar_ext can pull the helpers defined above
# without a circular import; it contributes ~50 more record generators + object/primitive
# ops, wired into the dispatch via the `_ext` reference used inside sample_task above.
from . import grammar_ext as _ext  # noqa: E402
ARRAY_GENERATORS = ARRAY_GENERATORS + _ext.RECORD_EXT

# v6 grammar extension: core-jq constructs (reduce/foreach/walk/paths/if/merge/INDEX)
# that v5 never emitted — see grammar_t5.py for the rationale and the try/catch exclusion.
from . import grammar_t5 as _t5  # noqa: E402
ARRAY_GENERATORS = ARRAY_GENERATORS + _t5.RECORD_T5
