"""Sample concrete JSON documents from a schema, with deterministic faker-style values."""
import random

FIRST = ["Alice", "Bob", "Carmen", "Diego", "Emma", "Farid", "Grace", "Hugo", "Iris",
         "Jonas", "Kira", "Liam", "Mona", "Nadia", "Omar", "Priya", "Quentin", "Rosa",
         "Sven", "Tara", "Umar", "Vera", "Wei", "Yuki", "Zoe"]
LAST = ["Martin", "Silva", "Chen", "Dubois", "Okafor", "Petrov", "Larsson", "Rossi",
        "Tanaka", "Novak", "Garcia", "Meyer", "Kaur", "Diallo", "Moreau"]
CITIES = ["Paris", "Lyon", "Berlin", "Tokyo", "Austin", "Toronto", "Lisbon", "Oslo",
          "Seoul", "Nairobi", "Bogota", "Prague", "Melbourne"]
WORDS = ["alpha", "harbor", "crimson", "delta", "ember", "falcon", "granite", "horizon",
         "indigo", "jasper", "kelp", "lumen", "meadow", "nickel", "onyx", "pixel",
         "quartz", "raven", "slate", "topaz"]


def _value(node: dict, rng: random.Random):
    t = node["t"]
    if t == "obj":
        return _object(node, rng)
    if t == "arr":
        lo, hi = node["n"]
        return [_value(node["item"], rng) for _ in range(rng.randint(lo, hi))]
    if t == "int":
        return rng.randint(node["lo"], node["hi"])
    if t == "float":
        return round(rng.uniform(node["lo"], node["hi"]), 2)
    if t == "bool":
        return rng.random() < 0.5
    if t == "enum":
        return rng.choice(node["vals"])
    if t == "str":
        k = node["kind"]
        if k == "name":
            return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if k == "email":
            return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}@{rng.choice(['example.com', 'mail.dev', 'corp.io'])}"
        if k == "word":
            return " ".join(rng.sample(WORDS, rng.randint(1, 3)))
        if k == "uuid":
            return "".join(rng.choice("0123456789abcdef") for _ in range(12))
        if k == "date":
            return f"{rng.randint(2023, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        if k == "city":
            return rng.choice(CITIES)
        if k == "url":
            return f"https://{rng.choice(WORDS)}.{rng.choice(['example.com', 'svc.local', 'dev.io'])}"
        if k == "sku":
            return f"{rng.choice('ABCDEFG')}{rng.randint(100, 999)}-{rng.choice(WORDS).upper()}"
        if k == "id":
            return f"{rng.choice(WORDS)}-{rng.randint(1, 999)}"
        if k == "path":
            return "/" + "/".join(rng.sample(WORDS, rng.randint(2, 4)))
        if k == "sentence":
            return " ".join(rng.sample(WORDS, rng.randint(3, 6)))
        if k == "numstr":  # a number encoded as a string, e.g. "42" (for tonumber)
            return str(rng.randint(1, 999))
        if k == "version":  # "v1.4.2" style (for ltrimstr("v"))
            return "v" + ".".join(str(rng.randint(0, 12)) for _ in range(rng.randint(2, 3)))
        if k == "filename":  # "report.pdf" style (for endswith filters)
            return f"{rng.choice(WORDS)}.{rng.choice(['py', 'txt', 'json', 'md', 'csv'])}"
    raise ValueError(f"unknown node {node}")


def _object(node: dict, rng: random.Random) -> dict:
    out = {}
    for name, child in node["fields"].items():
        if name in node.get("optional", []) and rng.random() < 0.35:
            continue  # optional field absent in this document
        out[name] = _value(child, rng)
    return out


def sample_documents(schema_info: dict, rng: random.Random, n_docs: int = 4) -> list:
    return [_value(schema_info["schema"], rng) for _ in range(n_docs)]
