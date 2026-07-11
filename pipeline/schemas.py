"""Synthetic schema generation over domain vocabularies.

Schema nodes:
  {"t": "obj", "fields": {name: node, ...}, "optional": [names...]}
  {"t": "arr", "item": node, "n": (lo, hi)}
  {"t": "int", "lo": int, "hi": int}
  {"t": "float", "lo": float, "hi": float}
  {"t": "bool"} | {"t": "enum", "vals": [...]}
  {"t": "str", "kind": one of name|email|word|uuid|date|city|url|sku|id}

v6: most field names are SYNTHESIZED per example from a combinatorial space (millions of
distinct names, each appearing a handful of times in the whole corpus), so memorizing the
field vocabulary is useless and the model must learn to COPY field names from the input.
v5's fixed 687-name vocabulary appeared ~2,800x per name — trivially memorizable, which is
why the v5 model scored 0.00 on the field-disjoint frozen benchmark. A bench-field
exclusion list keeps synthesized training names disjoint from every evaluation field.
"""
import itertools
import json
import random
from pathlib import Path

from .common import stable_hash

# Each domain: (record noun, field pool). Field pool entries: (name, leaf spec, p_optional)
DOMAINS = {
    "commerce": ("order", [
        ("id", {"t": "int", "lo": 1, "hi": 99999}, 0.0),
        ("sku", {"t": "str", "kind": "sku"}, 0.0),
        ("customer", {"t": "obj", "fields": {
            "name": {"t": "str", "kind": "name"},
            "email": {"t": "str", "kind": "email"}}, "optional": []}, 0.0),
        ("total", {"t": "float", "lo": 5, "hi": 900}, 0.0),
        ("quantity", {"t": "int", "lo": 1, "hi": 20}, 0.0),
        ("price", {"t": "float", "lo": 1, "hi": 250}, 0.0),
        ("status", {"t": "enum", "vals": ["paid", "pending", "refunded", "shipped"]}, 0.0),
        ("created_at", {"t": "str", "kind": "date"}, 0.0),
        ("discount", {"t": "float", "lo": 0, "hi": 50}, 0.5),
        ("gift", {"t": "bool"}, 0.3),
        ("coupon", {"t": "str", "kind": "id"}, 0.6),
    ]),
    "users": ("user", [
        ("id", {"t": "int", "lo": 1, "hi": 99999}, 0.0),
        ("username", {"t": "str", "kind": "id"}, 0.0),
        ("name", {"t": "str", "kind": "name"}, 0.0),
        ("email", {"t": "str", "kind": "email"}, 0.0),
        ("age", {"t": "int", "lo": 18, "hi": 90}, 0.0),
        ("city", {"t": "str", "kind": "city"}, 0.0),
        ("role", {"t": "enum", "vals": ["admin", "editor", "viewer", "owner"]}, 0.0),
        ("active", {"t": "bool"}, 0.0),
        ("last_login", {"t": "str", "kind": "date"}, 0.3),
        ("nickname", {"t": "str", "kind": "word"}, 0.6),
        ("followers", {"t": "int", "lo": 0, "hi": 50000}, 0.0),
    ]),
    "logs": ("log entry", [
        ("timestamp", {"t": "str", "kind": "date"}, 0.0),
        ("level", {"t": "enum", "vals": ["debug", "info", "warn", "error", "fatal"]}, 0.0),
        ("service", {"t": "enum", "vals": ["api", "auth", "worker", "db", "cache"]}, 0.0),
        ("message", {"t": "str", "kind": "word"}, 0.0),
        ("duration_ms", {"t": "int", "lo": 1, "hi": 30000}, 0.0),
        ("status_code", {"t": "enum", "vals": [200, 201, 301, 400, 404, 500, 503]}, 0.0),
        ("host", {"t": "str", "kind": "url"}, 0.0),
        ("trace_id", {"t": "str", "kind": "uuid"}, 0.4),
        ("retries", {"t": "int", "lo": 0, "hi": 8}, 0.3),
    ]),
    "ci": ("build", [
        ("build_id", {"t": "int", "lo": 100, "hi": 99999}, 0.0),
        ("branch", {"t": "str", "kind": "id"}, 0.0),
        ("commit", {"t": "str", "kind": "uuid"}, 0.0),
        ("result", {"t": "enum", "vals": ["success", "failure", "cancelled", "flaky"]}, 0.0),
        ("duration_sec", {"t": "int", "lo": 10, "hi": 7200}, 0.0),
        ("author", {"t": "str", "kind": "name"}, 0.0),
        ("tests_passed", {"t": "int", "lo": 0, "hi": 5000}, 0.0),
        ("tests_failed", {"t": "int", "lo": 0, "hi": 200}, 0.0),
        ("started_at", {"t": "str", "kind": "date"}, 0.0),
        ("cache_hit", {"t": "bool"}, 0.4),
    ]),
    "finance": ("transaction", [
        ("txn_id", {"t": "str", "kind": "uuid"}, 0.0),
        ("amount", {"t": "float", "lo": 0.5, "hi": 20000}, 0.0),
        ("currency", {"t": "enum", "vals": ["USD", "EUR", "GBP", "JPY"]}, 0.0),
        ("category", {"t": "enum", "vals": ["food", "travel", "salary", "rent", "utilities"]}, 0.0),
        ("account", {"t": "obj", "fields": {
            "iban": {"t": "str", "kind": "id"},
            "owner": {"t": "str", "kind": "name"}}, "optional": []}, 0.0),
        ("pending", {"t": "bool"}, 0.0),
        ("date", {"t": "str", "kind": "date"}, 0.0),
        ("fee", {"t": "float", "lo": 0, "hi": 45}, 0.4),
        ("note", {"t": "str", "kind": "word"}, 0.6),
    ]),
    "iot": ("reading", [
        ("sensor_id", {"t": "str", "kind": "id"}, 0.0),
        ("temperature", {"t": "float", "lo": -20, "hi": 45}, 0.0),
        ("humidity", {"t": "float", "lo": 0, "hi": 100}, 0.0),
        ("battery", {"t": "int", "lo": 0, "hi": 100}, 0.0),
        ("location", {"t": "obj", "fields": {
            "room": {"t": "str", "kind": "word"},
            "floor": {"t": "int", "lo": 0, "hi": 40}}, "optional": []}, 0.0),
        ("online", {"t": "bool"}, 0.0),
        ("recorded_at", {"t": "str", "kind": "date"}, 0.0),
        ("firmware", {"t": "str", "kind": "id"}, 0.4),
    ]),
    "repos": ("repository", [
        ("name", {"t": "str", "kind": "id"}, 0.0),
        ("owner", {"t": "str", "kind": "id"}, 0.0),
        ("stars", {"t": "int", "lo": 0, "hi": 200000}, 0.0),
        ("forks", {"t": "int", "lo": 0, "hi": 40000}, 0.0),
        ("language", {"t": "enum", "vals": ["Python", "Rust", "Go", "TypeScript", "C++"]}, 0.0),
        ("archived", {"t": "bool"}, 0.0),
        ("license", {"t": "enum", "vals": ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause"]}, 0.3),
        ("pushed_at", {"t": "str", "kind": "date"}, 0.0),
        ("open_issues", {"t": "int", "lo": 0, "hi": 900}, 0.0),
        ("homepage", {"t": "str", "kind": "url"}, 0.5),
    ]),
}


# Array-of-scalar / array-of-object fields injected into records so the model sees
# "the tags array contains X", "how many items in each order", "every score above 50",
# "distinct tags across posts" — real jq territory that record-of-scalars never exercised.
# These are ARR nodes, so leaf_paths() skips them (they never leak into scalar generators);
# only the array-aware generators in grammar_ext reach them via item["fields"].
_COLLECTION_FIELDS = [
    ("tags", {"t": "arr", "item": {"t": "enum",
              "vals": ["urgent", "new", "sale", "featured", "backorder", "fragile",
                       "priority", "clearance", "gift", "fresh"]}, "n": (1, 4)}),
    ("labels", {"t": "arr", "item": {"t": "enum",
                "vals": ["bug", "feature", "docs", "wontfix", "duplicate", "enhancement",
                         "regression", "blocked"]}, "n": (1, 4)}),
    ("roles", {"t": "arr", "item": {"t": "enum",
               "vals": ["admin", "editor", "viewer", "owner", "guest", "billing"]}, "n": (1, 3)}),
    ("scores", {"t": "arr", "item": {"t": "int", "lo": 0, "hi": 100}, "n": (2, 5)}),
    ("items", {"t": "arr", "item": {"t": "obj", "fields": {
                "sku": {"t": "str", "kind": "sku"}, "qty": {"t": "int", "lo": 1, "hi": 10}},
               "optional": []}, "n": (1, 5)}),
]
_TEXT_FIELDS = [
    ("description", {"t": "str", "kind": "sentence"}),
    ("title", {"t": "str", "kind": "sentence"}),
]


# ---------------------------------------------------------------------------
# Field-name COPY generalization (v5). The v4 model memorized its ~70 training field
# names and emitted them regardless of the actual input JSON (e.g. output `.message` on
# an input that only has {name,dept,salary}) — useless in the CLI. Fix: draw field names
# from a large typed vocabulary PLUS compose compound names (modifier+base), so the field
# vocabulary is far too large to memorize and the model must learn the general skill of
# copying the requested field name into the program. Types stay correct so documents.py
# still produces realistic values.
def _f(lo, hi): return {"t": "float", "lo": lo, "hi": hi}
def _i(lo, hi): return {"t": "int", "lo": lo, "hi": hi}
def _s(kind): return {"t": "str", "kind": kind}
def _e(vals): return {"t": "enum", "vals": vals}


_FLOAT_BASES = ["price", "cost", "amount", "total", "subtotal", "fee", "charge", "balance",
                "revenue", "salary", "budget", "fare", "rate", "deposit", "tax", "profit",
                "spend", "weight", "temperature", "humidity", "latitude", "longitude"]
_INT_BASES = ["quantity", "count", "stock", "units", "votes", "likes", "views", "shares",
              "followers", "comments", "replies", "age", "rank", "size", "duration",
              "latency", "distance", "capacity", "seats", "attempts", "retries", "score",
              "points", "level", "streak", "battery", "priority", "position"]
_STR_KINDS = {"name": ["name", "author", "owner", "customer", "manager", "assignee",
                       "creator", "contact", "recipient", "reviewer", "vendor", "supplier"],
              "word": ["title", "label", "subject", "headline", "keyword", "category_name"],
              "sentence": ["description", "note", "comment", "message", "bio", "summary"],
              "id": ["username", "handle", "slug", "code", "ref", "sku", "alias", "product_id",
                     "order_ref", "serial"],
              "email": ["email", "contact_email", "notify_email"],
              "city": ["city", "location", "region", "market", "branch", "hometown"],
              "date": ["created_at", "updated_at", "date", "timestamp", "due_date",
                       "published_at", "expires_at", "start_date", "end_date"],
              "url": ["url", "link", "homepage", "website", "endpoint", "avatar"],
              "uuid": ["uuid", "trace_id", "commit", "token", "session_id", "request_id"]}
_BOOL_NAMES = ["active", "enabled", "verified", "archived", "featured", "pinned", "public",
               "paid", "shipped", "urgent", "flagged", "gift", "online", "premium", "trial",
               "subscribed", "completed", "starred"]
_ENUM_FIELDS = {
    "status": ["paid", "pending", "refunded", "shipped", "processing", "cancelled",
               "delivered", "returned", "on_hold", "backordered", "failed", "completed",
               "draft", "authorized", "void", "fulfilled"],
    "level": ["debug", "info", "warn", "error", "fatal", "trace", "notice", "critical"],
    "result": ["success", "failure", "cancelled", "flaky", "passed", "timeout", "skipped",
               "aborted", "errored", "queued", "running"],
    "role": ["admin", "editor", "viewer", "owner", "guest", "member", "moderator",
             "contributor", "maintainer", "support", "developer"],
    "currency": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL"],
    "category": ["food", "travel", "salary", "rent", "utilities", "groceries",
                 "entertainment", "healthcare", "transport", "insurance", "education"],
    "language": ["Python", "Rust", "Go", "TypeScript", "Java", "Ruby", "JavaScript",
                 "Kotlin", "Swift", "Scala", "Elixir", "Haskell", "C", "PHP"],
    "priority": ["low", "medium", "high", "critical", "urgent", "trivial"],
    "plan": ["free", "basic", "pro", "team", "business", "enterprise"],
    "team": ["red", "blue", "green", "alpha", "beta", "gamma", "north", "south"],
    "region": ["us-east", "us-west", "eu-central", "ap-south", "sa-east"],
    "department": ["eng", "sales", "marketing", "finance", "hr", "ops", "legal", "design"],
}
_FLOAT_MODS = ["unit", "total", "net", "gross", "avg", "max", "min", "base", "list", "final"]
_INT_MODS = ["daily", "weekly", "monthly", "total", "max", "min", "avg", "current", "peak"]


# ---------------------------------------------------------------------------
# v6: per-example field-name SYNTHESIS. Compose names from word pools (plus occasional
# syllable-synthesized stems and digit suffixes) — the space is tens of millions, so a
# given name recurs ~1-5 times across a 2M-example corpus and recall cannot beat copying.
# The pools deliberately avoid the v5 vocabulary's words.
_SYN_A = [
    "arrival", "backfill", "berth", "billing", "bin", "bracket", "burst", "cadence",
    "carton", "census", "chamber", "checkpoint", "cohort", "console", "courier", "crate",
    "cycle_end", "delta_run", "dispatch", "dock", "draft_pick", "elevation", "enrollment",
    "escrow", "expiry", "fleet", "gateway_hop", "hatch", "haul", "incline", "ingest",
    "intake_row", "junction", "kiln", "lane", "ledger_row", "lot", "manifest_row",
    "meridian", "milepost", "node_hop", "orbit", "outflow", "parcel_row", "payout_run",
    "pickup", "pipeline_leg", "pivot", "plot_row", "presale", "quadrant", "quarry",
    "relay", "renewal", "roster_row", "rotation", "runway_leg", "settle", "shipment_leg",
    "shift_block", "silo", "sortie", "spool", "sprint_leg", "stint", "stockpile",
    "strand", "surge", "tarmac", "terminal_leg", "threshold_row", "tranche", "transit_leg",
    "trench", "turnstile", "uplink", "vault_row", "voyage", "waypoint", "wharf", "yield_run",
]
_SYN_B = [
    "aggregate", "allotment", "apex", "ballast", "bandwidth", "bond", "buffer", "burden",
    "caliber", "cargo_mass", "ceiling_val", "clearance", "coefficient", "consumption",
    "credit_line", "damping", "density_val", "displacement", "drag", "drawdown",
    "efficiency", "elasticity", "emission", "exposure", "flux", "footprint", "friction",
    "gradient_val", "headroom", "impedance", "inertia", "influx", "intensity", "leakage",
    "leverage", "load_factor", "magnitude_val", "momentum", "occupancy", "offset_val",
    "onset", "overhead_val", "payload_mass", "persistence", "premium_val", "pressure_val",
    "proximity", "recoil", "residual", "resonance", "runoff", "saturation", "slack",
    "span_val", "spread_val", "stiffness", "surplus", "tension", "tolerance_val",
    "torque", "traction", "turnover_val", "utilization", "variance_val", "velocity_val",
    "viscosity", "voltage", "wastage", "wattage", "yield_val",
]
_SYN_SUFFIX = ["", "", "", "_a", "_b", "_x2", "_q3", "_v2", "_r1", "_lo", "_hi", "_mid"]
_SYL_ON = ["br", "cl", "dr", "fl", "gr", "kr", "pl", "sk", "sl", "sp", "st", "tr", "vl", "zh"]
_SYL_NUC = ["a", "e", "i", "o", "u", "au", "ei", "ou"]
_SYL_COD = ["b", "d", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "z"]


def _load_bench_exclusions():
    """Field names used by ANY evaluation split (frozen public, sealed, dev-novel) must
    never be synthesized into training data — that would erode the benchmarks' field-
    disjointness guarantee for models trained on v6+. Returns (exact set, prefix set).

    Prefers the prebuilt names-only file (bench/frozen/bench_field_exclusions.json) so
    datagen hosts (GPU pods) only ever receive extracted field NAMES — never sealed
    items. Rebuild it with:  python -m pipeline.schemas  (writes the file)."""
    root = Path(__file__).resolve().parent.parent
    names = set()
    prebuilt = root / "bench" / "frozen" / "bench_field_exclusions.json"
    if prebuilt.exists():
        names = set(json.loads(prebuilt.read_text())["names"])
    else:
        for rel in ("bench/frozen/nl2jq-bench-1.0.0.jsonl",
                    "bench/sealed/nl2jq-bench-sealed-v1.jsonl",
                    "bench/devnovel/devnovel-v1.jsonl"):
            fp = root / rel
            if not fp.exists():
                continue
            for line in fp.open():
                it = json.loads(line)
                stack = [it["input"]]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        names.update(node.keys())
                        stack.extend(node.values())
                    elif isinstance(node, list):
                        stack.extend(node)
    prefixes = set()
    for n in names:
        for ln in range(4, len(n) + 1):
            prefixes.add(n[:ln])
    return names, prefixes


_BENCH_EXCL_NAMES, _BENCH_EXCL_PREFIXES = _load_bench_exclusions()


def _collides_with_bench(name: str) -> bool:
    if name in _BENCH_EXCL_NAMES:
        return True
    if name in _BENCH_EXCL_PREFIXES:          # name is a prefix of a bench field
        return True
    for ln in range(4, len(name) + 1):        # a bench field is a prefix of name
        if name[:ln] in _BENCH_EXCL_NAMES:
            return True
    return False


# v7: components come from a LARGE English pool (~6k words), not the small curated lists.
# WHY (learned from the v6 failure): with ~150 components, BPE gave every component its
# own token and the model's copy circuit only worked over those familiar word-tokens —
# real-world field names (cultivar, tail_no) tokenized into rare char fragments it could
# not emit. With ~6k components the training subword distribution covers the pieces real
# field names are made of, so copying generalizes at the token level. Name-level bench
# exclusion still keeps every actual evaluation field name out of training.
_WORDPOOL = json.loads(
    (Path(__file__).resolve().parent / "wordpool_v7.json").read_text())["words"]


def synth_field_name(rng: random.Random) -> str:
    """One synthesized field name from a multi-million space, bench-disjoint."""
    for _ in range(20):  # collisions with the exclusion list are rare; retry a few times
        r = rng.random()
        if r < 0.45:      # two-part combo from the big pool
            name = f"{rng.choice(_WORDPOOL)}_{rng.choice(_WORDPOOL)}"
        elif r < 0.60:    # single word (very common in real JSON)
            name = rng.choice(_WORDPOOL)
        elif r < 0.72:    # abbreviation-style truncation, like real devs write (qty, ltr)
            w = rng.choice(_WORDPOOL)
            cut = w[: rng.randint(2, 4)]
            name = cut if rng.random() < 0.4 else f"{rng.choice(_WORDPOOL)}_{cut}"
        elif r < 0.85:    # three-part / suffixed compound
            name = (f"{rng.choice(_WORDPOOL)}_{rng.choice(_WORDPOOL)}"
                    + (rng.choice(_SYN_SUFFIX) or "_" + rng.choice(_WORDPOOL)))
        else:             # syllable-synthesized stem (pure char-fragment copying practice)
            stem = "".join(rng.choice(p) for p in (_SYL_ON, _SYL_NUC, _SYL_COD))
            stem += "".join(rng.choice(p) for p in (_SYL_NUC, _SYL_COD)) if rng.random() < 0.6 else ""
            name = stem if rng.random() < 0.5 else f"{stem}_{rng.choice(_WORDPOOL)}"
        name += rng.choice(_SYN_SUFFIX) if rng.random() < 0.15 else ""
        if not _collides_with_bench(name):
            return name
    return f"zz_{rng.randrange(10**6)}"  # guaranteed-safe fallback


def _synth_typed_field(rng: random.Random):
    """(name, leaf spec) with a synthesized name and a type drawn like _typed_field."""
    name = synth_field_name(rng)
    r = rng.random()
    if r < 0.35:
        return name, _f(1, rng.choice([100, 900, 5000]))
    if r < 0.70:
        return name, _i(0, rng.choice([20, 100, 5000, 200000]))
    if r < 0.85:
        return name, _s(rng.choice(["word", "id", "name", "date"]))
    if r < 0.95:
        return name, {"t": "bool"}
    # enum with per-example synthesized values (a fixed value pool would just recreate
    # the value-memorization problem one level down)
    vals = []
    while len(vals) < rng.randint(3, 4):
        v = ("".join(rng.choice(p) for p in (_SYL_ON, _SYL_NUC, _SYL_COD))
             + "".join(rng.choice(p) for p in (_SYL_NUC, _SYL_COD)))
        if v not in vals:
            vals.append(v)
    return name, _e(vals)


def _typed_field(rng):
    """One (name, leaf_spec), sometimes a compound name — a large, hard-to-memorize space."""
    r = rng.random()
    if r < 0.30:
        base = rng.choice(_FLOAT_BASES)
        name = f"{rng.choice(_FLOAT_MODS)}_{base}" if rng.random() < 0.4 else base
        return name, _f(1, rng.choice([100, 900, 5000]))
    if r < 0.58:
        base = rng.choice(_INT_BASES)
        name = f"{rng.choice(_INT_MODS)}_{base}" if rng.random() < 0.4 else base
        return name, _i(0, rng.choice([20, 100, 5000, 200000]))
    if r < 0.74:
        kind = rng.choice(list(_STR_KINDS))
        return rng.choice(_STR_KINDS[kind]), _s(kind)
    if r < 0.86:
        return rng.choice(_BOOL_NAMES), {"t": "bool"}
    name = rng.choice(list(_ENUM_FIELDS))
    return name, _e(_ENUM_FIELDS[name])


def sample_record_schema(rng: random.Random, domain: str) -> dict:
    """Build a record schema: a few domain-signature fields (realism + bench coverage)
    mixed with fields drawn from the large typed vocabulary (field-name copy generalization).

    With some probability, inject a collection field (array of scalars/objects) and/or a
    free-text sentence field for the array-aware and string-splitting generators."""
    _, pool = DOMAINS[domain]
    fields, optional = {}, []
    # 1-3 domain-signature fields keep some coherence and the bench's domain-specific names
    for name, leaf, p_opt in rng.sample(pool, min(rng.randint(1, 3), len(pool))):
        fields[name] = leaf
        if rng.random() < p_opt:
            optional.append(name)
    # 3-6 typed fields: mostly SYNTHESIZED unique names (copy-forcing, v6), with a
    # ~30% share of the common vocabulary so everyday names (price, status...) stay strong
    for _ in range(rng.randint(3, 6)):
        name, leaf = (_synth_typed_field(rng) if rng.random() < 0.7
                      else _typed_field(rng))
        fields.setdefault(name, leaf)
    if rng.random() < 0.5:
        cname, cleaf = rng.choice(_COLLECTION_FIELDS)
        fields.setdefault(cname, cleaf)
    if rng.random() < 0.3:
        tname, tleaf = rng.choice(_TEXT_FIELDS)
        fields.setdefault(tname, tleaf)
    return {"t": "obj", "fields": fields, "optional": optional}


# Homogeneous top-level objects: a settings blob (all numbers) or a feature-flag blob
# (all booleans). These exercise map_values, to_entries|select(.value), object-values
# aggregation — none of which the mixed-type record objects could reach.
_SETTINGS_KEYS = ["width", "height", "timeout", "retries", "volume", "limit", "threshold",
                  "max_connections", "cache_size", "port", "workers", "brightness",
                  "font_size", "page_size", "quality", "margin"]
_FLAG_KEYS = ["dark_mode", "notifications", "autosave", "beta", "telemetry", "compact_view",
              "verbose", "sound", "wifi", "bluetooth", "sync", "analytics", "auto_update",
              "spellcheck", "location", "backups"]


def sample_settings_schema(rng: random.Random) -> dict:
    keys = rng.sample(_SETTINGS_KEYS, rng.randint(3, 6))
    # v6: swap ~half the keys for synthesized ones (novel settings names must be copyable)
    keys = [synth_field_name(rng) if rng.random() < 0.5 else k for k in keys]
    fields = {k: {"t": "int", "lo": 1, "hi": 200} for k in keys}
    return {"schema": {"t": "obj", "fields": fields, "optional": []},
            "noun": "setting", "domain": "settings_num",
            "family": stable_hash({"settings": sorted(keys)})}


def sample_flags_schema(rng: random.Random) -> dict:
    keys = rng.sample(_FLAG_KEYS, rng.randint(3, 6))
    keys = [synth_field_name(rng) if rng.random() < 0.5 else k for k in keys]
    fields = {k: {"t": "bool"} for k in keys}
    return {"schema": {"t": "obj", "fields": fields, "optional": []},
            "noun": "flag", "domain": "flags_bool",
            "family": stable_hash({"flags": sorted(keys)})}


# Primitive (non-record) top-level shapes: arrays of scalars, nested arrays, bare
# strings. These are common in real jq use ("sum these numbers", "join these tags",
# "split this path") and were missing from the record-only distribution, so the model
# used to collapse on them. Each carries a noun for NL phrasing.
_NUM_ITEMS = [
    ({"t": "int", "lo": 1, "hi": 100}, "number"),
    ({"t": "int", "lo": 0, "hi": 5000}, "value"),
    ({"t": "float", "lo": 0.5, "hi": 500}, "amount"),
    ({"t": "int", "lo": 1, "hi": 20}, "score"),
]
_STR_ITEMS = [
    ({"t": "str", "kind": "word"}, "word"),
    ({"t": "str", "kind": "name"}, "name"),
    ({"t": "str", "kind": "city"}, "city"),
    ({"t": "str", "kind": "id"}, "tag"),
    ({"t": "str", "kind": "path"}, "path"),
    ({"t": "str", "kind": "version"}, "version"),
    ({"t": "str", "kind": "filename"}, "file"),
]


def sample_primitive_schema(rng: random.Random) -> dict:
    """A non-record top-level shape: number array, string array, nested number array,
    a single string, a numeric-string array (tonumber), or key/value pairs (from_entries).
    Returns the same dict shape as sample_schema."""
    kind = rng.choices(["num_arr", "str_arr", "nested", "scalar_str", "numstr", "kvpairs"],
                       weights=[4, 3, 2, 2, 1.3, 1])[0]
    if kind == "num_arr":
        item, noun = rng.choice(_NUM_ITEMS)
        top = {"t": "arr", "item": item, "n": (3, 8)}
        fam = f"prim:num:{item['t']}"
    elif kind == "str_arr":
        item, noun = rng.choice(_STR_ITEMS)
        top = {"t": "arr", "item": item, "n": (3, 7)}
        fam = f"prim:str:{item['kind']}"
    elif kind == "nested":
        item, noun = rng.choice(_NUM_ITEMS[:2]), "group"
        top = {"t": "arr", "item": {"t": "arr", "item": item[0], "n": (2, 4)}, "n": (2, 5)}
        noun = "group"
        fam = "prim:nested"
    elif kind == "numstr":
        top = {"t": "arr", "item": {"t": "str", "kind": "numstr"}, "n": (3, 6)}
        noun = "value"
        fam = "prim:numstr"
    elif kind == "kvpairs":
        top = {"t": "arr", "item": {"t": "obj", "fields": {
                "key": {"t": "str", "kind": "word"}, "value": {"t": "int", "lo": 1, "hi": 200}},
               "optional": []}, "n": (3, 5)}
        noun = "pair"
        fam = "prim:kvpairs"
    else:  # scalar_str
        item, noun = rng.choice([({"t": "str", "kind": "path"}, "path"),
                                 ({"t": "str", "kind": "sentence"}, "text")])
        top = item
        fam = f"prim:scalar:{item['kind']}"
    return {"schema": top, "noun": noun, "domain": "primitive",
            "family": stable_hash({"prim": fam})}


def sample_schema(rng: random.Random) -> dict:
    """Top-level schema: mostly array-of-records / single object, plus ~22% primitive
    (scalar arrays, nested arrays, bare strings) and ~14% homogeneous objects
    (settings/flags) so the model isn't record-only."""
    r = rng.random()
    if r < 0.22:
        return sample_primitive_schema(rng)
    if r < 0.30:
        return sample_settings_schema(rng)
    if r < 0.36:
        return sample_flags_schema(rng)
    domain = rng.choice(list(DOMAINS.keys()))
    record = sample_record_schema(rng, domain)
    noun = DOMAINS[domain][0]
    if rng.random() < 0.65:
        top = {"t": "arr", "item": record, "n": (3, 8)}
    else:
        top = dict(record)
        if rng.random() < 0.6:  # embed a nested array of records from another domain
            sub_domain = rng.choice(list(DOMAINS.keys()))
            sub_noun = DOMAINS[sub_domain][0]
            field_name = {"order": "orders", "user": "members", "log entry": "events",
                          "build": "builds", "transaction": "transactions",
                          "reading": "readings", "repository": "repos"}[sub_noun]
            top = {"t": "obj",
                   "fields": {**top["fields"],
                              field_name: {"t": "arr",
                                           "item": sample_record_schema(rng, sub_domain),
                                           "n": (3, 7)}},
                   "optional": top["optional"]}
            top["_arr_field"] = (field_name, sub_noun)
    return {"schema": top, "noun": noun, "domain": domain,
            "family": stable_hash({"d": domain, "f": sorted(_field_names(top))})}


def _field_names(node, prefix="") -> list:
    if node["t"] == "obj":
        out = []
        for k, v in node["fields"].items():
            out.append(prefix + k)
            out.extend(_field_names(v, prefix + k + "."))
        return out
    if node["t"] == "arr":
        return _field_names(node["item"], prefix + "[].")
    return []
