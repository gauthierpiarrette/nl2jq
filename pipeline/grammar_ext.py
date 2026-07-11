"""v4 grammar extension: broad feature coverage + bounded composition + CONTRASTIVE
phrasings. Program shapes were validated against pinned jq during design; every emitted
example is still execution-verified by generate.py, so a generator that produces a
bad/degenerate program is simply dropped.

All NL phrasings are authored in-repo (no model-generated text enters the dataset).

Contrastive design: paired generators share a near-identical NL stem but diverge in one
trigger word, mapping to different programs — teaching which words select which op:
  "the highest price"                 -> map(.price) | max        (a value)
  "the order with the highest price"  -> max_by(.price)           (a record)
  "top 3 orders by total"             -> sort_by(-.total)|.[0:3]
  "top 3 orders by total, ids only"   -> ... | map(.id)
  "orders where paid"                 -> map(select(.paid))
  "the ids of orders where paid"      -> map(select(.paid)|.id)
"""

from .grammar import (_by_type, _fmt_num, _values_at, path_phrase, sample_condition)


# ---- slot pickers (ctx.paths holds 3-tuples: (path, node, is_optional)) ----
def _pick(ctx, types):
    c = _by_type(ctx.paths, types)          # -> [(path, node)]
    return ctx.rng.choice(c) if c else None


def _fields1(ctx, types=None):
    """Top-level (len-1) leaves as [(name, node)], optionally type-filtered."""
    return [(p[0], n) for p, n, _o in ctx.paths
            if len(p) == 1 and (types is None or n["t"] in types)]


def _names1(ctx):
    return list(dict.fromkeys(p[0] for p, _n, _o in ctx.paths if len(p) == 1))


def _array_fields(ctx, item_types):
    """Record fields that are arrays with the given item type(s): [(name, arr_node)]."""
    out = []
    for k, v in ctx.item.get("fields", {}).items():
        if v.get("t") == "arr" and v.get("item", {}).get("t") in item_types:
            out.append((k, v))
    return out


def _nested_leaves(ctx):
    return [(p, n) for p, n, _o in ctx.paths if len(p) >= 2]


def _text_field(ctx):
    """A string field whose values are multi-word (good for split(' ')|length)."""
    cands = [(p, n) for p, n, _o in ctx.paths
             if len(p) == 1 and n["t"] == "str" and n.get("kind") in ("sentence", "name")]
    return ctx.rng.choice(cands) if cands else None


def _plur(s):
    """Plural-ish phrase without producing 'statuss'/'starss' — leave s/x endings alone."""
    return s if s.endswith(("s", "x")) else s + "s"


_ID_LIKE = ("id", "name", "sku", "key", "username", "build_id", "txn_id", "sensor_id",
            "commit", "branch", "email")


def _keep_field(ctx, names):
    """Prefer an identifier-ish field for the 'keep' slot (matches bench convention)."""
    for cand in _ID_LIKE:
        if cand in names:
            return cand
    return ctx.rng.choice(names)


def _median(ctx, path):
    vals = sorted(v for v in _values_at(ctx.records, path) if isinstance(v, (int, float)))
    return _fmt_num(vals[len(vals) // 2]) if vals else None


def _rand_n(ctx, lo=2, hi=3):
    return ctx.rng.randint(lo, min(hi, max(lo, len(ctx.records) - 1)))


# ================================ RESHAPE ==================================
def gx_project(ctx):
    """map({a}), map({a,b}) — keep a subset of fields (contrast: full record)."""
    names = _names1(ctx)
    if len(names) < 2:
        return None
    k = ctx.rng.choice([1, 2, 2, 3]) if len(names) >= 3 else 2
    picked = ctx.rng.sample(names, min(k, len(names)))
    keys = ", ".join(picked)
    human = " and ".join(h.replace("_", " ") for h in picked)
    P = ctx.prefix
    return {"program": f"{P}map({{{keys}}})", "tier": 2, "tags": ["reshape"],
            "nl": [f"just {human} for each {ctx.noun}", f"keep only {human} in every {ctx.noun}",
                   f"reduce each {ctx.noun} to {human}", f"pick {human} from each {ctx.noun}",
                   f"each {ctx.noun} with only {human}"]}


def gx_rename(ctx):
    """map({new: .old, keep: .keep}) — rename one field, keep another."""
    names = _names1(ctx)
    if len(names) < 2:
        return None
    old, keep = ctx.rng.sample(names, 2)
    new = ctx.rng.choice(["renamed", "label", "key", old + "_id"])
    oh, kh, P = old.replace("_", " "), keep.replace("_", " "), ctx.prefix
    return {"program": f"{P}map({{{new}: .{old}, {keep}: .{keep}}})", "tier": 3,
            "tags": ["reshape"],
            "nl": [f"rename {oh} to {new} in each {ctx.noun}, keeping {kh}",
                   f"relabel {oh} as {new} for every {ctx.noun}",
                   f"change the {oh} key to {new}, keep {kh}"]}


def gx_construct_join(ctx):
    """map({keep, name: (.a + " " + .b)}) — keep a field, add a joined one. [seed-085]"""
    strs = _fields1(ctx, ("str", "enum"))
    keepable = _names1(ctx)
    if len(strs) < 2 or len(keepable) < 3:
        return None
    (a, _), (b, _) = ctx.rng.sample(strs, 2)
    others = [n for n in keepable if n not in (a, b)]
    if not others:
        return None
    k = _keep_field(ctx, others)
    new = ctx.rng.choice(["full", "label", "combined", "display"])
    ah, bh, kh, P = a.replace("_", " "), b.replace("_", " "), k.replace("_", " "), ctx.prefix
    return {"program": f'{P}map({{{k}, {new}: (.{a} + " " + .{b})}})', "tier": 3,
            "tags": ["reshape", "string"],
            "nl": [f"each {ctx.noun} as {kh} and a {new} joining {ah} and {bh}",
                   f"keep {kh} and add {new} built from {ah} plus {bh}",
                   f"each record as {kh} and {new} where {new} joins {ah} and {bh}"]}


def gx_flag(ctx):
    """map({keep, flag: (.n > k)}) — add a boolean flag. [seed-100]"""
    num = _pick(ctx, ("int", "float"))
    names = _names1(ctx)
    if not num or not names:
        return None
    (npath, _) = num
    thr = _median(ctx, npath)
    if thr is None:
        return None
    k = _keep_field(ctx, [n for n in names if (npath[0],) != (n,)] or names)
    flag = ctx.rng.choice(["expensive", "high", "large", "over_limit", "flagged"])
    nh, kh, P = path_phrase(npath), k.replace("_", " "), ctx.prefix
    return {"program": f"{P}map({{{k}, {flag}: (.{npath[0]} > {thr})}})", "tier": 3,
            "tags": ["reshape", "boolean"],
            "nl": [f"each {ctx.noun} as {kh} and whether {nh} is over {thr} as {flag}",
                   f"add a {flag} flag (true when {nh} exceeds {thr}) to each {ctx.noun}, keep {kh}",
                   f"each {ctx.noun}'s {kh} plus a {flag} flag for {nh} above {thr}"]}


def gx_del_field(ctx):
    names = _names1(ctx)
    if len(names) < 2:
        return None
    drop = ctx.rng.choice(names)
    ph, P = drop.replace("_", " "), ctx.prefix
    return {"program": f"{P}map(del(.{drop}))", "tier": 2, "tags": ["del", "reshape"],
            "nl": [f"remove the {ph} field from each {ctx.noun}",
                   f"drop {ph} from every {ctx.noun}", f"each {ctx.noun} without {ph}",
                   f"strip out {ph} from the {ctx.plural}"]}


def gx_nested_pluck(ctx):
    """map(.location.room) — pluck a nested field. [seed-026]"""
    nested = _nested_leaves(ctx)
    if not nested:
        return None
    path, _ = ctx.rng.choice(nested)
    last, parent, P = path[-1].replace("_", " "), path[-2].replace("_", " "), ctx.prefix
    return {"program": f"{P}map(.{'.'.join(path)})", "tier": 2, "tags": ["pluck", "nested"],
            "nl": [f"each {ctx.noun}'s {parent} {last}", f"the {last} of every {parent}",
                   f"pull out the {parent} {last} from each {ctx.noun}"]}


# ============================ STRING TRANSFORMS ============================
def gx_field_case(ctx):
    """map(.name | ascii_upcase) — case a field (contrast: bare string array). [seed-019]"""
    c = _pick(ctx, ("str", "enum"))
    if not c:
        return None
    path, _ = c
    fn, w = ctx.rng.choice([("ascii_upcase", ["uppercased", "in uppercase", "in capitals"]),
                            ("ascii_downcase", ["lowercased", "in lowercase"])])
    ph, ww, P = path_phrase(path), ctx.rng.choice(w), ctx.prefix
    return {"program": f"{P}map(.{'.'.join(path)} | {fn})", "tier": 2, "tags": ["string"],
            "nl": [f"the {ph} of every {ctx.noun}, {ww}", f"each {ph} {ww}", f"every {ph} {ww}"]}


def gx_field_concat(ctx):
    """map(.first + " " + .last) — join two fields. [seed-075]"""
    strs = _fields1(ctx, ("str", "enum"))
    if len(strs) < 2:
        return None
    (a, _), (b, _) = ctx.rng.sample(strs, 2)
    ah, bh, P = a.replace("_", " "), b.replace("_", " "), ctx.prefix
    return {"program": f'{P}map(.{a} + " " + .{b})', "tier": 2, "tags": ["string", "concat"],
            "nl": [f"{ah} and {bh} joined with a space for each {ctx.noun}",
                   f"each {ctx.noun}'s {ah} then {bh} as one string",
                   f"combine {ah} and {bh} into a single value per {ctx.noun}"]}


def gx_field_split(ctx):
    """map(.email | split("@")[1]) — split a field, take a part. [seed-088]"""
    c = _pick(ctx, ("str",))
    if not c:
        return None
    path, _ = c
    vals = [v for v in _values_at(ctx.records, path) if isinstance(v, str)]
    sep = next((s for s in ("@", "/", ".", "-") if vals and all(s in v for v in vals)), None)
    if not sep:
        return None
    idx = ctx.rng.choice([0, 1])
    which = "first" if idx == 0 else "second"
    ph, P = path_phrase(path), ctx.prefix
    tail = (f"each {ph} up to the '{sep}'" if idx == 0 else f"each {ph} after the '{sep}'")
    nls = [f"the {which} part of each {ph} split on '{sep}'", tail,
           f"split each {ph} by '{sep}' and take the {which} piece"]
    if sep == "@" and idx == 1:
        nls.append(f"the domain of each {ph} (the part after the @)")
    return {"program": f'{P}map(.{path[0]} | split("{sep}")[{idx}])', "tier": 3,
            "tags": ["string", "split"], "nl": nls}


def gx_field_gsub(ctx):
    """map(.title | gsub(" "; "_")) — substitute within a field. [seed-042]"""
    c = _text_field(ctx) or _pick(ctx, ("str", "enum"))
    if not c:
        return None
    path, _ = c
    frm, to, w = ctx.rng.choice([(" ", "_", "spaces with underscores"),
                                 ("-", " ", "dashes with spaces"),
                                 (" ", "-", "spaces with dashes")])
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f'{P}map(.{".".join(path)} | gsub("{frm}"; "{to}"))', "tier": 3,
            "tags": ["string", "gsub"],
            "nl": [f"each {ph} with {w}", f"replace {w} in every {ph}",
                   f"every {ph}, swapping {w}"]}


def gx_field_wordcount(ctx):
    """map(.description | split(" ") | length) — words per sentence field. [seed-092]"""
    c = _text_field(ctx)
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f'{P}map(.{".".join(path)} | split(" ") | length)', "tier": 3,
            "tags": ["string", "split", "count"],
            "nl": [f"the number of words in each {ph}", f"how many words each {ph} has",
                   f"word count of every {ph}"]}


# ============================ FILTER EXTENSIONS ===========================
def _str_pred(ctx):
    """(jq_predicate_body, [bare nl clauses], ph) for a string-field predicate, or None."""
    c = _pick(ctx, ("str", "enum"))
    if not c:
        return None
    path, _ = c
    vals = [v for v in _values_at(ctx.records, path) if isinstance(v, str) and v]
    if not vals:
        return None
    v = ctx.rng.choice(vals)
    ph, jp = path_phrase(path), ".".join(path)
    kind = ctx.rng.choice(["contains", "startswith", "endswith"])
    if kind == "contains":
        sub = v[: max(1, len(v) // 2)]
        return (f'.{jp} | contains("{sub}")',
                [f"{ph} contains {sub}", f"{ph} has {sub} in it", f"{ph} includes {sub}"], ph)
    if kind == "startswith":
        sub = v[:1]
        return (f'.{jp} | startswith("{sub}")',
                [f"{ph} starts with {sub}", f"{ph} begins with {sub}"], ph)
    sub = v[-3:] if v[-3:] else v
    return (f'.{jp} | endswith("{sub}")',
            [f"{ph} ends with {sub}", f"{ph} ends in {sub}"], ph)


def gx_filter_str(ctx):
    """map(select(.f | contains/startswith/endswith)). Contrast: +pluck. [seed-034,059]"""
    p = _str_pred(ctx)
    if not p:
        return None
    body, nls, _ph = p
    nl, P = ctx.rng.choice, ctx.prefix
    return {"program": f"{P}map(select({body}))", "tier": 2, "tags": ["filter", "string"],
            "nl": [f"{ctx.plural} where {nl(nls)}", f"only {ctx.plural} whose {nl(nls)}",
                   f"keep {ctx.plural} where {nl(nls)}"]}


def gx_filter_str_pluck(ctx):
    """map(select(.name | startswith("A")) | .name) — filter then pluck. [seed-033]"""
    p = _str_pred(ctx)
    tgt = _pick(ctx, ("str", "enum", "int", "float"))
    if not p or not tgt:
        return None
    body, nls, _ph = p
    tp, _ = tgt
    th, nl, P = path_phrase(tp), ctx.rng.choice, ctx.prefix
    return {"program": f"{P}map(select({body}) | .{'.'.join(tp)})", "tier": 3,
            "tags": ["filter", "string", "pluck"],
            "nl": [f"the {th}s of {ctx.plural} where {nl(nls)}",
                   f"{th} for {ctx.plural} whose {nl(nls)}"]}


def gx_filter_test(ctx):
    """map(select(.f | test("@"))) — regex test (contrast with contains). [seed-057,095]"""
    c = _pick(ctx, ("str",))
    if not c:
        return None
    path, _ = c
    vals = [v for v in _values_at(ctx.records, path) if isinstance(v, str)]
    pat, desc = None, None
    if vals and all("@" in v for v in vals):
        pat, desc = "@", ["contain an @", "have an @ in them"]
    elif vals and all(" " in v for v in vals):
        pat, desc = " ", ["contain a space", "have a space in them"]
    if not pat:
        return None
    ph, nl, P = path_phrase(path), ctx.rng.choice, ctx.prefix
    return {"program": f'{P}map(select(.{".".join(path)} | test("{pat}")))', "tier": 3,
            "tags": ["filter", "regex"],
            "nl": [f"{ctx.plural} whose {ph} {nl(desc)}", f"{ctx.plural} where {ph} {nl(desc)}"]}


def gx_filter_bool_compound(ctx):
    """map(select(A and/or B)). [seed-046,047]"""
    a, b = sample_condition(ctx), sample_condition(ctx)
    if not a or not b:
        return None
    (ja, na), (jb, nb) = a, b
    ja, jb = ja[len("select("):-1], jb[len("select("):-1]   # unwrap; re-parenthesize
    if ja == jb:
        return None
    op, word = ctx.rng.choice([("and", "and"), ("or", "or")])
    nl, P = ctx.rng.choice, ctx.prefix
    return {"program": f"{P}map(select(({ja}) {op} ({jb})))", "tier": 3,
            "tags": ["filter", "boolean"],
            "nl": [f"{ctx.plural} where {nl(na)} {word} {nl(nb)}",
                   f"{ctx.plural} that are {nl(na)} {word} {nl(nb)}"]}


def gx_filter_between(ctx):
    """map(select(.n >= lo and .n <= hi)). [seed-065]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    vals = sorted(v for v in _values_at(ctx.records, path) if isinstance(v, (int, float)))
    if len(vals) < 3:
        return None
    lo, hi = _fmt_num(vals[len(vals) // 4]), _fmt_num(vals[3 * len(vals) // 4])
    ph, jp, P = path_phrase(path), ".".join(path), ctx.prefix
    return {"program": f"{P}map(select(.{jp} >= {lo} and .{jp} <= {hi}))", "tier": 3,
            "tags": ["filter"],
            "nl": [f"{ctx.plural} where {ph} is between {lo} and {hi}",
                   f"{ctx.plural} with {ph} from {lo} to {hi} inclusive",
                   f"{ctx.plural} whose {ph} is {lo} to {hi}"]}


def gx_filter_not(ctx):
    """map(select(.flag | not)). [seed-062]"""
    c = _pick(ctx, ("bool",))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}map(select(.{'.'.join(path)} | not))", "tier": 2,
            "tags": ["filter", "boolean"],
            "nl": [f"{ctx.plural} that are not {ph}", f"{ctx.plural} where {ph} is false",
                   f"the {ctx.plural} without {ph}"]}


def gx_first_match(ctx):
    """first(.[] | select(cond)). [seed-061]"""
    cond = sample_condition(ctx)
    if not cond:
        return None
    jc, nls = cond
    nl, P = ctx.rng.choice, ctx.prefix
    return {"program": f"{P}first(.[] | {jc})", "tier": 3, "tags": ["filter", "first"],
            "nl": [f"the first {ctx.noun} where {nl(nls)}",
                   f"the first {ctx.noun} that is {nl(nls)}",
                   f"find the first {ctx.noun} with {nl(nls)}"]}


def gx_any(ctx):
    """any(.n > k) / all(...). [seed-015]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    thr = _median(ctx, path)
    if thr is None:
        return None
    fn = ctx.rng.choice(["any", "all"])
    ph, jp, P = path_phrase(path), ".".join(path), ctx.prefix
    if fn == "any":
        nls = [f"does any {ctx.noun} have {ph} over {thr}",
               f"is there a {ctx.noun} with {ph} above {thr}",
               f"do any {ctx.plural} exceed {thr} in {ph}"]
    else:
        nls = [f"do all {ctx.plural} have {ph} over {thr}",
               f"is every {ctx.noun}'s {ph} above {thr}"]
    return {"program": f"{P}{fn}(.{jp} > {thr})", "tier": 2, "tags": ["quantifier"], "nl": nls}


# ============================== AGGREGATES ================================
def gx_count_distinct(ctx):
    """[.[].status] | unique | length. [seed-067]"""
    c = _pick(ctx, ("enum", "str", "int"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}[.[].{'.'.join(path)}] | unique | length", "tier": 2,
            "tags": ["unique", "count"],
            "nl": [f"how many distinct {_plur(ph)} are there", f"the number of unique {_plur(ph)}",
                   f"count of different {ph} values"]}


def gx_cond_sum(ctx):
    """[.[] | select(cond) | .n] | add. [seed-022,079]"""
    cond = sample_condition(ctx)
    num = _pick(ctx, ("int", "float"))
    if not cond or not num:
        return None
    jc, nls = cond
    (npath, _) = num
    nh, nl, P = path_phrase(npath), ctx.rng.choice, ctx.prefix
    return {"program": f"{P}[.[] | {jc} | .{'.'.join(npath)}] | add", "tier": 3,
            "tags": ["filter", "sum"],
            "nl": [f"total {nh} of {ctx.plural} where {nl(nls)}",
                   f"sum of {nh} for {ctx.plural} where {nl(nls)}",
                   f"add up {nh} across {ctx.plural} where {nl(nls)}"]}


def gx_alt_field(ctx):
    """map(.price // 0) — default missing to 0 (contrast: +| add). [seed-023,064]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    ph, jp, P = path_phrase(path), ".".join(path), ctx.prefix
    if ctx.rng.random() < 0.5:
        return {"program": f"{P}map(.{jp} // 0)", "tier": 2, "tags": ["alternative"],
                "nl": [f"the {ph} of each {ctx.noun}, defaulting missing ones to 0",
                       f"each {ph}, treating missing as zero",
                       f"{ph} or 0 when absent, per {ctx.noun}"]}
    return {"program": f"{P}map(.{jp} // 0) | add", "tier": 3, "tags": ["alternative", "sum"],
            "nl": [f"total {ph}, treating missing as zero",
                   f"sum of {ph} counting absent values as 0",
                   f"add up {ph}, missing ones default to zero"]}


def gx_pluck_array(ctx):
    """[.[] | select(cond) | .id] — filtered field as a JSON array. [seed-074]"""
    cond = sample_condition(ctx)
    tgt = _pick(ctx, ("str", "enum", "int", "float"))
    if not cond or not tgt:
        return None
    jc, nls = cond
    tp, _ = tgt
    th, nl, P = path_phrase(tp), ctx.rng.choice, ctx.prefix
    return {"program": f"{P}[.[] | {jc} | .{'.'.join(tp)}]", "tier": 2, "tags": ["filter", "pluck"],
            "nl": [f"the {th}s of {ctx.plural} where {nl(nls)}, as an array",
                   f"a JSON array of {th} for {ctx.plural} where {nl(nls)}",
                   f"collect {th} from {ctx.plural} where {nl(nls)}"]}


def gx_percentage(ctx):
    """(map(select(cond)) | length) * 100 / length. [seed-083]"""
    cond = sample_condition(ctx)
    if not cond:
        return None
    jc, nls = cond
    nl, P = ctx.rng.choice, ctx.prefix
    return {"program": f"{P}(map({jc}) | length) * 100 / length", "tier": 4,
            "tags": ["count", "arith"],
            "nl": [f"what percentage of {ctx.plural} are {nl(nls)}",
                   f"the percent of {ctx.plural} where {nl(nls)}"]}


def gx_above_avg(ctx):
    """(map(.n) | add / length) as $avg | map(select(.n > $avg)). [seed-089]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    ph, jp, P = path_phrase(path), ".".join(path), ctx.prefix
    return {"program": (f"{P}(map(.{jp}) | add / length) as $avg | "
                        f"map(select(.{jp} > $avg))"), "tier": 4, "tags": ["filter", "avg"],
            "nl": [f"the {ctx.plural} whose {ph} is above the average {ph}",
                   f"{ctx.plural} with above-average {ph}",
                   f"{ctx.plural} where {ph} beats the mean {ph}"]}


# ============================ SORT / SLICE ================================
def gx_sort_desc_slice(ctx):
    """sort_by(-.n) | .[0:k] (+ map(.f)). Contrast: ascending. [seed-025]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    n, ph, P = _rand_n(ctx), path_phrase(path), ctx.prefix
    if ctx.rng.random() < 0.5:
        return {"program": f"{P}sort_by(-.{'.'.join(path)}) | .[0:{n}]", "tier": 3,
                "tags": ["sort", "slice"],
                "nl": [f"the top {n} {ctx.plural} by {ph}",
                       f"the {n} {ctx.plural} with the highest {ph}",
                       f"the {n} highest-{ph} {ctx.plural}"]}
    tgt = _pick(ctx, ("str", "enum", "int", "float"))
    if not tgt:
        return None
    tp, _ = tgt
    th = path_phrase(tp)
    return {"program": f"{P}sort_by(-.{'.'.join(path)}) | .[0:{n}] | map(.{'.'.join(tp)})",
            "tier": 3, "tags": ["sort", "slice", "pluck"],
            "nl": [f"the {th}s of the top {n} {ctx.plural} by {ph}",
                   f"the {n} highest-{ph} {ctx.plural}, {th} only",
                   f"just the {th} of the {n} biggest {ctx.plural} by {ph}"]}


def gx_sort_asc_slice(ctx):
    """sort_by(.n) | .[0:k] | map(.f) — the k smallest, projected. [seed-093]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    n, ph, P = _rand_n(ctx), path_phrase(path), ctx.prefix
    tgt = _pick(ctx, ("str", "enum", "int", "float"))
    if not tgt:
        return None
    tp, _ = tgt
    th = path_phrase(tp)
    return {"program": f"{P}sort_by(.{'.'.join(path)}) | .[0:{n}] | map(.{'.'.join(tp)})",
            "tier": 3, "tags": ["sort", "slice", "pluck"],
            "nl": [f"the {th}s of the {n} {ctx.plural} with the lowest {ph}",
                   f"the {n} cheapest {ctx.plural} by {ph}, {th} only",
                   f"just the {th} of the {n} smallest {ctx.plural} by {ph}"]}


def gx_offset_slice(ctx):
    """.[a:b] — a mid-list slice (contrast: .[0:n] "first n"). [seed-016]"""
    m = len(ctx.records)
    if m < 3:
        return None
    a = ctx.rng.randint(1, max(1, m - 2))
    b = min(m, a + ctx.rng.choice([1, 2]))
    P = ctx.prefix
    nls = [f"{ctx.plural} {a + 1} through {b}", f"the slice from index {a} to {b}"]
    if (a, b) == (1, 3):
        nls.append(f"the second and third {ctx.plural}")
    return {"program": f"{P}.[{a}:{b}]", "tier": 2, "tags": ["slice"], "nl": nls}


def gx_nth_extreme(ctx):
    """[.[].score] | sort | reverse | .[k] — the k-th highest value. [seed-082]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    k = ctx.rng.choice([1, 2])
    ordn = {1: "second", 2: "third"}[k]
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}[.[].{'.'.join(path)}] | sort | reverse | .[{k}]", "tier": 3,
            "tags": ["sort", "index"],
            "nl": [f"the {ordn} highest {ph}", f"the {ordn} largest {ph} value"]}


# ============================== GROUP-BY ==================================
def _group_key(ctx):
    cats = _fields1(ctx, ("enum", "str"))
    return ctx.rng.choice(cats) if cats else None


def gx_group_sum(ctx):
    """group_by(.cat) | map({cat, n: (map(.num)|add)}) (+ top-N, + {key,value}). [seed-011,076,094]"""
    cat, num = _group_key(ctx), _pick(ctx, ("int", "float"))
    if not cat or not num:
        return None
    ck, _ = cat
    (nk, _) = num
    nj = ".".join(nk)
    ch, nh, P = ck.replace("_", " "), path_phrase(nk), ctx.prefix
    roll = ctx.rng.random()
    if roll < 0.3:                    # top-N groups by total
        n = _rand_n(ctx)
        base = f"{P}group_by(.{ck}) | map({{{ck}: .[0].{ck}, {nk[-1]}: (map(.{nj}) | add)}})"
        return {"program": f"{base} | sort_by(-.{nk[-1]}) | .[0:{n}]", "tier": 4,
                "tags": ["group_by", "sum", "sort", "slice"],
                "nl": [f"the top {n} {ch}s by total {nh}",
                       f"the {n} {ch}s with the highest total {nh}, as {{{ch}, {nk[-1]}}}"]}
    if roll < 0.5:                    # explicit {key, value} object shape
        return {"program": (f"{P}group_by(.{ck}) | "
                            f"map({{key: .[0].{ck}, value: (map(.{nj}) | add)}})"),
                "tier": 3, "tags": ["group_by", "sum"], "order_insensitive": True,
                "nl": [f"total {nh} per {ch} as {{key, value}} objects",
                       f"{nh} summed by {ch}, each as a key/value object"]}
    return {"program": f"{P}group_by(.{ck}) | map({{{ck}: .[0].{ck}, {nk[-1]}: (map(.{nj}) | add)}})",
            "tier": 3, "tags": ["group_by", "sum"], "order_insensitive": True,
            "nl": [f"total {nh} per {ch}", f"sum of {nh} for each {ch}",
                   f"{nh} totals grouped by {ch}"]}


def gx_group_count_filter(ctx):
    """group_by(.cat) | map({cat, count: length}) (+ select count > 1). [seed-018,077]"""
    cat = _group_key(ctx)
    if not cat:
        return None
    ck, _ = cat
    ch, P = ck.replace("_", " "), ctx.prefix
    base = f"{P}group_by(.{ck}) | map({{{ck}: .[0].{ck}, count: length}})"
    if ctx.rng.random() < 0.4:
        return {"program": f"{base} | map(select(.count > 1))", "tier": 4,
                "tags": ["group_by", "count", "filter"], "order_insensitive": True,
                "nl": [f"count of {ctx.plural} per {ch}, keeping only {ch}s with more than one",
                       f"{ch}s that have more than one {ctx.noun}, with their counts"]}
    return {"program": base, "tier": 3, "tags": ["group_by", "count"], "order_insensitive": True,
            "nl": [f"how many {ctx.plural} for each {ch}", f"count of {ctx.plural} per {ch}",
                   f"number of {ctx.plural} grouped by {ch}"]}


def gx_group_avg(ctx):
    """group_by(.cat) | map({cat, avg: ((map(.num)|add)/length)}). [seed-069]"""
    cat, num = _group_key(ctx), _pick(ctx, ("int", "float"))
    if not cat or not num:
        return None
    ck, _ = cat
    (nk, _) = num
    ch, nh, P = ck.replace("_", " "), path_phrase(nk), ctx.prefix
    return {"program": (f"{P}group_by(.{ck}) | "
                        f"map({{{ck}: .[0].{ck}, avg: ((map(.{'.'.join(nk)}) | add) / length)}})"),
            "tier": 4, "tags": ["group_by", "arith"], "order_insensitive": True,
            "nl": [f"average {nh} per {ch}", f"mean {nh} for each {ch}",
                   f"the average {nh} grouped by {ch}"]}


def gx_group_collect(ctx):
    """group_by(.cat) | map({cat, ids: map(.id)}). [seed-081]"""
    cat = _group_key(ctx)
    tgt = _pick(ctx, ("int", "str", "enum"))
    if not cat or not tgt:
        return None
    ck, _ = cat
    tp, _ = tgt
    tj = ".".join(tp)
    ch, th, P = ck.replace("_", " "), path_phrase(tp), ctx.prefix
    return {"program": (f"{P}group_by(.{ck}) | "
                        f"map({{{ck}: .[0].{ck}, {tp[-1]}s: map(.{tj})}})"),
            "tier": 4, "tags": ["group_by", "collect"], "order_insensitive": True,
            "nl": [f"group {ctx.plural} by {ch} and list the {th}s in each group",
                   f"for each {ch}, the {th}s of its {ctx.plural}"]}


# ========================= NUMBER / EXTREME ==============================
def gx_field_round(ctx):
    """map(.score | round/floor/ceil). [seed-044]"""
    c = _pick(ctx, ("float", "int"))
    if not c:
        return None
    path, _ = c
    fn, w = ctx.rng.choice([("round", "rounded to the nearest integer"),
                            ("floor", "rounded down"), ("ceil", "rounded up")])
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}map(.{'.'.join(path)} | {fn})", "tier": 2, "tags": ["number"],
            "nl": [f"each {ph} {w}", f"every {ph}, {w}", f"the {ph}s {w}"]}


def gx_field_scale(ctx):
    """map(.price * 1.08). [seed-045]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    factor = ctx.rng.choice(["1.08", "1.2", "1.1", "0.9", "2"])
    pct = {"1.08": "8 percent", "1.2": "20 percent", "1.1": "10 percent"}.get(factor)
    ph, P = path_phrase(path), ctx.prefix
    nls = [f"each {ph} multiplied by {factor}", f"{ph} scaled by {factor} for each {ctx.noun}"]
    if pct:
        nls.append(f"each {ph} with {pct} added" if float(factor) > 1
                   else f"each {ph} reduced by {pct}")
    return {"program": f"{P}map(.{'.'.join(path)} * {factor})", "tier": 2, "tags": ["arith"],
            "nl": nls}


def gx_avg_field(ctx):
    """(map(.age) | add) / length. [seed-009]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}(map(.{'.'.join(path)}) | add) / length", "tier": 2,
            "tags": ["arith", "avg"],
            "nl": [f"the average {ph}", f"the mean {ph}", f"average {ph} across the {ctx.plural}"]}


def gx_field_extreme(ctx):
    """map(.temp) | max — the extreme VALUE (contrast: max_by -> the record). [seed-051]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    fn, w = ctx.rng.choice([("max", ["highest", "largest", "maximum", "greatest"]),
                            ("min", ["lowest", "smallest", "minimum"])])
    ph, ww, P = path_phrase(path), ctx.rng.choice(w), ctx.prefix
    return {"program": f"{P}map(.{'.'.join(path)}) | {fn}", "tier": 2, "tags": [fn],
            "nl": [f"the {ww} {ph}", f"the {ww} {ph} value", f"what is the {ww} {ph}"]}


def gx_minmax_obj(ctx):
    """{min: (map(.price)|min), max: (map(.price)|max)}. [seed-096]"""
    c = _pick(ctx, ("int", "float"))
    if not c:
        return None
    path, _ = c
    jp, ph, P = ".".join(path), path_phrase(path), ctx.prefix
    return {"program": f"{P}{{min: (map(.{jp}) | min), max: (map(.{jp}) | max)}}", "tier": 3,
            "tags": ["object", "min", "max"],
            "nl": [f"the minimum and maximum {ph} as {{min, max}}",
                   f"the lowest and highest {ph} in one object", f"min and max {ph}"]}


def gx_update_field(ctx):
    """map(.stock += 1). [seed-071]"""
    c = _pick(ctx, ("int",))
    if not c:
        return None
    path, _ = c
    delta = ctx.rng.choice([1, 5, 10])
    op, word = ctx.rng.choice([("+=", "increase"), ("-=", "decrease")])
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}map(.{'.'.join(path)} {op} {delta})", "tier": 3, "tags": ["update"],
            "nl": [f"{word} every {ctx.noun}'s {ph} by {delta}",
                   f"{word} {ph} by {delta} for each {ctx.noun}"]}


def gx_unique_by(ctx):
    """unique_by(.email). [seed-030]"""
    c = _pick(ctx, ("str", "enum"))
    if not c:
        return None
    path, _ = c
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}unique_by(.{'.'.join(path)})", "tier": 3, "tags": ["unique_by"],
            "order_insensitive": True,
            "nl": [f"the {ctx.plural} deduplicated by {ph}", f"one {ctx.noun} per {ph}",
                   f"{ctx.plural} with duplicate {ph} removed"]}


def gx_interp(ctx):
    r'''map("\(.name): \(.age)"). [seed-024]'''
    names = _names1(ctx)
    if len(names) < 2:
        return None
    a, b = ctx.rng.sample(names, 2)
    sep = ctx.rng.choice([": ", " - ", " = "])
    ah, bh, P = a.replace("_", " "), b.replace("_", " "), ctx.prefix
    return {"program": f'{P}map("\\(.{a}){sep}\\(.{b})")', "tier": 4, "tags": ["string", "interp"],
            "nl": [f"a '{ah}{sep}{bh}' string for each {ctx.noun}",
                   f"format each {ctx.noun} as {ah}{sep.strip()} {bh}",
                   f"each {ctx.noun} rendered '{ah}{sep}{bh}'"]}


# ============================== @csv / @tsv ==============================
def gx_csv(ctx):
    """[.[].id] | @csv  or  @tsv. [seed-013]"""
    c = _pick(ctx, ("str", "enum", "int", "float"))
    if not c:
        return None
    path, _ = c
    fmt, w = ctx.rng.choice([("@csv", "comma separated"), ("@tsv", "tab separated")])
    ph, P = path_phrase(path), ctx.prefix
    return {"program": f"{P}[.[].{'.'.join(path)}] | {fmt}", "tier": 3, "tags": ["format"],
            "nl": [f"the {_plur(ph)} as a {w} line", f"all {ph} values on one {w} row",
                   f"{ph} values joined into a {w} string"]}


# ========================= ARRAY-FIELD OPS ===============================
def _arr_scalar_field(ctx):
    fs = _array_fields(ctx, ("str", "enum"))
    return ctx.rng.choice(fs) if fs else None


def _arr_value(ctx, name):
    vals = []
    for r in ctx.records:
        a = r.get(name)
        if isinstance(a, list):
            vals.extend(x for x in a if isinstance(x, str))
    return ctx.rng.choice(vals) if vals else None


def gx_array_contains(ctx):
    """map(select(.tags | index("x"))) (+ | .name). [seed-020,078]"""
    f = _arr_scalar_field(ctx)
    if not f:
        return None
    name, _ = f
    val = _arr_value(ctx, name)
    if not val:
        return None
    nh, P = name.replace("_", " "), ctx.prefix
    if ctx.rng.random() < 0.5:
        tgt = _pick(ctx, ("str", "enum"))
        if tgt:
            tp, _ = tgt
            th = path_phrase(tp)
            return {"program": f'{P}map(select(.{name} | index("{val}")) | .{".".join(tp)})',
                    "tier": 3, "tags": ["filter", "array", "pluck"],
                    "nl": [f"the {th}s of {ctx.plural} whose {nh} contains {val}",
                           f"{th} for {ctx.plural} tagged {val}"]}
    return {"program": f'{P}map(select(.{name} | index("{val}")))', "tier": 3,
            "tags": ["filter", "array"],
            "nl": [f"{ctx.plural} whose {nh} contains {val}", f"{ctx.plural} tagged {val}",
                   f"{ctx.plural} where {nh} includes {val}"]}


def gx_array_len(ctx):
    """map(.items | length) — size of an array field. [seed-063]"""
    fs = _array_fields(ctx, ("str", "enum", "int", "float", "obj"))
    if not fs:
        return None
    name, _ = ctx.rng.choice(fs)
    nh, P = name.replace("_", " "), ctx.prefix
    return {"program": f"{P}map(.{name} | length)", "tier": 2, "tags": ["array", "count"],
            "nl": [f"how many {nh} each {ctx.noun} has", f"the number of {nh} per {ctx.noun}",
                   f"the count of {nh} in each {ctx.noun}"]}


def gx_array_all(ctx):
    """map(select(.scores | all(. > k))). [seed-087]"""
    fs = _array_fields(ctx, ("int", "float"))
    if not fs:
        return None
    name, node = ctx.rng.choice(fs)
    lo, hi = node["item"].get("lo", 0), node["item"].get("hi", 100)
    thr = (lo + hi) // 2
    single = name[:-1] if name.endswith("s") else name
    nh, P = name.replace("_", " "), ctx.prefix
    return {"program": f"{P}map(select(.{name} | all(. > {thr})))", "tier": 4,
            "tags": ["filter", "array", "quantifier"],
            "nl": [f"{ctx.plural} where every {single.replace('_', ' ')} is above {thr}",
                   f"{ctx.plural} whose {nh} are all over {thr}",
                   f"{ctx.plural} with all {nh} greater than {thr}"]}


def gx_array_flatten_unique(ctx):
    """[.[].tags[]] | unique — distinct values across all array fields. [seed-091]"""
    f = _arr_scalar_field(ctx)
    if not f:
        return None
    name, _ = f
    single = name[:-1] if name.endswith("s") else name
    nh, P = name.replace("_", " "), ctx.prefix
    return {"program": f"{P}[.[].{name}[]] | unique", "tier": 3, "tags": ["array", "unique"],
            "nl": [f"all distinct {nh} across the {ctx.plural}",
                   f"the unique set of {nh} used by any {ctx.noun}",
                   f"every distinct {single.replace('_', ' ')} across all {ctx.plural}"]}


def gx_array_frequency(ctx):
    """[.[].tags[]] | group_by(.) | map({tag: .[0], n: length}). [seed-086]"""
    f = _arr_scalar_field(ctx)
    if not f:
        return None
    name, _ = f
    single = name[:-1] if name.endswith("s") else name
    P = ctx.prefix
    return {"program": (f"{P}[.[].{name}[]] | group_by(.) | "
                        f"map({{{single}: .[0], n: length}})"), "tier": 4,
            "tags": ["array", "group_by", "count"], "order_insensitive": True,
            "nl": [f"how many times each {single.replace('_', ' ')} appears across all {ctx.plural}",
                   f"the frequency of each {single.replace('_', ' ')} over the {ctx.plural}"]}


# Weighted list appended to ARRAY_GENERATORS by grammar.py.
RECORD_EXT = [
    (gx_project, 1.5), (gx_rename, 0.7), (gx_construct_join, 0.8), (gx_flag, 0.8),
    (gx_del_field, 0.9), (gx_nested_pluck, 1.1),
    (gx_field_case, 1.2), (gx_field_concat, 0.9), (gx_field_split, 0.9),
    (gx_field_gsub, 0.8), (gx_field_wordcount, 0.6),
    (gx_filter_str, 1.4), (gx_filter_str_pluck, 1.0), (gx_filter_test, 0.8),
    (gx_filter_bool_compound, 1.2), (gx_filter_between, 0.9), (gx_filter_not, 0.8),
    (gx_first_match, 0.9), (gx_any, 0.9),
    (gx_count_distinct, 1.1), (gx_cond_sum, 1.1), (gx_alt_field, 1.0),
    (gx_pluck_array, 1.1), (gx_percentage, 0.6), (gx_above_avg, 0.8),
    (gx_sort_desc_slice, 1.4), (gx_sort_asc_slice, 0.9), (gx_offset_slice, 0.7),
    (gx_nth_extreme, 0.9),
    (gx_group_sum, 1.5), (gx_group_count_filter, 1.2), (gx_group_avg, 1.0),
    (gx_group_collect, 0.9),
    (gx_field_round, 0.9), (gx_field_scale, 0.9), (gx_avg_field, 1.0),
    (gx_field_extreme, 1.2), (gx_minmax_obj, 0.8),
    (gx_update_field, 0.7), (gx_unique_by, 0.9), (gx_interp, 0.9), (gx_csv, 0.9),
    (gx_array_contains, 1.1), (gx_array_len, 0.9), (gx_array_all, 0.7),
    (gx_array_flatten_unique, 0.9), (gx_array_frequency, 0.8),
]


# ======================= OBJECT-TOP GENERATORS ===========================
def _weighted_obj(choices, rng):
    tasks, weights = zip(*choices)
    return rng.choices(tasks, weights=weights)[0]


def sample_settings_task(schema_info, docs, rng):
    """Homogeneous all-numeric object (settings blob)."""
    doc0 = docs[0]
    keys = list(doc0.keys())
    if not keys:
        return None
    k = rng.choice(keys)
    kh = k.replace("_", " ")
    mv_factor = rng.choice([2, 2, 3, 10])
    mv_nl = (["double every value in the object", "each value doubled", "twice every value"]
             if mv_factor == 2 else
             [f"every value multiplied by {mv_factor}", f"each setting scaled by {mv_factor}",
              f"multiply every value by {mv_factor}"])
    ch = [
        ({"program": f"map_values(. * {mv_factor})", "tags": ["object", "map_values"],
          "nl": mv_nl}, 1.4),
        ({"program": "to_entries", "tags": ["entries", "object"],
          "nl": ["the settings as key/value pairs", "list each setting as a key and value",
                 "turn the object into key/value entries"]}, 1.2),
        ({"program": "[.[]]", "tags": ["values", "object"],
          "nl": ["all the values in the settings object", "just the values",
                 "every setting's value"]}, 1.0),
        ({"program": "[.[]] | add / length", "tags": ["avg", "object"],
          "nl": ["the average of the values in the object", "the mean setting value",
                 "average across all the values"]}, 1.0),
        ({"program": "to_entries | max_by(.value) | .key", "tags": ["entries", "max"],
          "nl": ["the key with the highest value", "which setting has the largest value",
                 "the name of the biggest setting"]}, 1.0),
        ({"program": "to_entries | min_by(.value) | .key", "tags": ["entries", "min"],
          "nl": ["the key with the smallest value", "which setting is lowest"]}, 0.7),
        ({"program": "[.[]] | max", "tags": ["max", "object"],
          "nl": ["the largest value in the object", "the maximum setting value"]}, 0.8),
        ({"program": f".{k}", "tags": ["path"],
          "nl": [f"the {kh} setting", f"what is {kh}", f"get {kh}"]}, 0.8),
    ]
    t = _weighted_obj(ch, rng)
    return {"program": t["program"], "tier": 3, "tags": t["tags"], "nl": t["nl"]}


def sample_flags_task(schema_info, docs, rng):
    """Homogeneous all-boolean object (feature-flag blob)."""
    doc0 = docs[0]
    keys = list(doc0.keys())
    if not keys:
        return None
    k = rng.choice(keys)
    kh = k.replace("_", " ")
    ch = [
        ({"program": "to_entries | map(select(.value) | .key)", "tags": ["entries", "filter"],
          "nl": ["the names of settings that are enabled", "which flags are on",
                 "the keys whose value is true", "the features that are turned on"]}, 1.5),
        ({"program": "to_entries | map(select(.value | not) | .key)", "tags": ["entries", "filter"],
          "nl": ["the flags that are off", "which settings are disabled",
                 "the keys whose value is false"]}, 1.0),
        ({"program": "[.[]] | map(select(.)) | length", "tags": ["count", "object"],
          "nl": ["how many flags are enabled", "the number of settings turned on",
                 "count of true values"]}, 1.0),
        ({"program": "to_entries", "tags": ["entries", "object"],
          "nl": ["the flags as key/value pairs", "each flag as a key and value"]}, 1.0),
        ({"program": "[.[]]", "tags": ["values", "object"],
          "nl": ["all the values in the object", "just the flag values"]}, 0.8),
        ({"program": f".{k}", "tags": ["path"],
          "nl": [f"is {kh} enabled", f"the {kh} flag", f"what is {kh} set to"]}, 0.8),
    ]
    t = _weighted_obj(ch, rng)
    return {"program": t["program"], "tier": 3, "tags": t["tags"], "nl": t["nl"]}


def go_to_entries(schema_info, docs, rng):
    return {"program": "to_entries", "tier": 2, "tags": ["entries", "object"],
            "nl": ["the object as key/value pairs", "list the fields as key/value entries",
                   "turn this object into key/value pairs"]}


def go_values(schema_info, docs, rng):
    return {"program": "[.[]]", "tier": 2, "tags": ["values", "object"],
            "nl": ["all the values in the object", "just the values",
                   "the object's values as a list"]}


def go_obj_length(schema_info, docs, rng):
    return {"program": "length", "tier": 1, "tags": ["length", "object"], "constant_ok": True,
            "nl": ["how many fields does this object have", "the number of fields in the object",
                   "count of keys in the object"]}


OBJECT_GENERIC = [(go_to_entries, 1.0), (go_values, 1.0), (go_obj_length, 0.8)]


# ====================== PRIMITIVE EXTENSIONS =============================
def gp_numstr(doc0, rng):
    """Array of numeric strings: tonumber then aggregate. [seed-060]"""
    if not (isinstance(doc0, list) and doc0
            and all(isinstance(x, str) and x.lstrip("-").isdigit() for x in doc0)):
        return None
    ch = [
        ({"program": "map(tonumber) | add", "tags": ["tonumber", "sum"],
          "nl": ["convert the string amounts to numbers and sum them",
                 "parse each as a number and total them", "sum them as numbers"]}, 1.6),
        ({"program": "map(tonumber)", "tags": ["tonumber"],
          "nl": ["convert each string to a number", "parse them all as numbers",
                 "the values as numbers"]}, 1.0),
        ({"program": "map(tonumber) | max", "tags": ["tonumber", "max"],
          "nl": ["the largest, as a number", "parse them and take the maximum"]}, 0.8),
        ({"program": "map(tonumber) | add / length", "tags": ["tonumber", "avg"],
          "nl": ["the average of the numeric strings", "parse and average them"]}, 0.8),
    ]
    t = rng.choices([c for c, _ in ch], weights=[w for _, w in ch])[0]
    return {"program": t["program"], "tier": 3, "tags": t["tags"], "nl": t["nl"]}


def gp_kvpairs(doc0, rng):
    """Array of {key, value} objects: from_entries and friends. [seed-041]"""
    if not (isinstance(doc0, list) and doc0
            and all(isinstance(x, dict) and "key" in x and "value" in x for x in doc0)):
        return None
    ch = [
        ({"program": "from_entries", "tags": ["from_entries"],
          "nl": ["turn the list of key/value pairs into an object",
                 "build an object from these key/value pairs",
                 "collapse the pairs into one object"]}, 1.6),
        ({"program": "map(.value)", "tags": ["pluck"],
          "nl": ["just the values from each pair", "the value of every pair"]}, 0.9),
        ({"program": "map(.key)", "tags": ["pluck"],
          "nl": ["just the keys", "the key of each pair"]}, 0.9),
        ({"program": "map(.value) | add", "tags": ["sum"],
          "nl": ["the total of all the values", "sum the values across the pairs"]}, 0.9),
        ({"program": "max_by(.value) | .key", "tags": ["max", "pluck"],
          "nl": ["the key of the pair with the highest value",
                 "which key has the biggest value"]}, 0.8),
    ]
    t = rng.choices([c for c, _ in ch], weights=[w for _, w in ch])[0]
    return {"program": t["program"], "tier": 3, "tags": t["tags"], "nl": t["nl"]}


def prim_num_ext(doc0, rng):
    """Extra number-array shapes not in gp_num: offset slice, {min,max}, squares-sum, nth."""
    nums = [v for v in doc0 if isinstance(v, (int, float))]
    if len(nums) < 3:
        return None
    ch = []
    a = rng.randint(1, max(1, len(nums) - 2))
    b = min(len(nums), a + rng.choice([1, 2]))
    slice_nl = [f"items {a + 1} through {b}", f"the slice from index {a} to {b}"]
    if (a, b) == (1, 3):
        slice_nl.append("the second and third items")
    ch.append(({"program": f".[{a}:{b}]", "tags": ["slice"], "nl": slice_nl}, 1.4))
    ch.append(({"program": "{min: min, max: max}", "tags": ["object", "min", "max"],
                "nl": ["the minimum and maximum as {min, max}", "min and max in one object",
                       "the smallest and largest together"]}, 1.0))
    ch.append(({"program": "map(. * .) | add", "tags": ["arith", "sum"],
                "nl": ["the sum of the squares", "add up each number squared",
                       "total of the squared values"]}, 1.2))
    if any(float(v) != int(v) for v in nums):
        fn, w = rng.choice([("round", "rounded"), ("floor", "rounded down"),
                            ("ceil", "rounded up")])
        ch.append(({"program": f"map({fn})", "tags": ["number"],
                    "nl": [f"each number {w}", f"the numbers {w}"]}, 1.0))
    k = rng.choice([1, 2])
    ch.append(({"program": f"sort | reverse | .[{k}]", "tags": ["sort", "index"],
                "nl": [f"the {'second' if k == 1 else 'third'} largest number",
                       f"the {'second' if k == 1 else 'third'} highest value"]}, 1.0))
    t = rng.choices([c for c, _ in ch], weights=[w for _, w in ch])[0]
    return {"program": t["program"], "tier": 2, "tags": t["tags"], "nl": t["nl"]}


def prim_str_ext(doc0, rng):
    """Extra string-array shapes: reverse-each, ltrimstr, endswith filter, index."""
    strs = [v for v in doc0 if isinstance(v, str)]
    if len(strs) < 2:
        return None
    ch = [({"program": "map(explode | reverse | implode)", "tags": ["string"],
            "nl": ["each string reversed", "reverse every string", "the strings reversed"]}, 1.2),
          ({"program": "unique | length", "tags": ["unique", "count"],
            "nl": ["how many distinct values", "the number of unique values",
                   "count of distinct entries"]}, 1.0)]
    if all(s and s[0] == strs[0][0] for s in strs) and len(strs[0]) > 1:
        pre = strs[0][0]
        ch.append(({"program": f'map(ltrimstr("{pre}"))', "tags": ["string"],
                    "nl": [f"strip the leading '{pre}' from each",
                           f"remove the '{pre}' prefix from every value"]}, 1.3))
    exts = {s.rsplit(".", 1)[-1] for s in strs if "." in s}
    if len(exts) > 1:
        ext = sorted(exts)[0]
        ch.append(({"program": f'map(select(endswith(".{ext}")))', "tags": ["filter", "string"],
                    "nl": [f"only the ones ending in .{ext}", f"the values that end with .{ext}",
                           f"just the .{ext} files"]}, 1.3))
    val = rng.choice(strs)
    ch.append(({"program": f'index("{val}")', "tags": ["index"],
                "nl": [f"the position of '{val}' in the list", f"where '{val}' is",
                       f"the index of '{val}'"]}, 1.0))
    t = rng.choices([c for c, _ in ch], weights=[w for _, w in ch])[0]
    return {"program": t["program"], "tier": 2, "tags": t["tags"], "nl": t["nl"]}
