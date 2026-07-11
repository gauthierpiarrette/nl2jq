"""Run frontier-model zero-shot baselines against nl2jq-bench (SPEC §4).

These numbers are the comparison row in the results table. They are measurements only —
frontier outputs are NEVER added to the training set (that would raise ToS/licensing
questions for a published dataset).

    # needs ANTHROPIC_API_KEY (and/or OPENAI_API_KEY) in env — source ~/.nl2jq_secrets
    python -m bench.baselines --provider anthropic
    python -m bench.baselines --provider openai --model gpt-5

Note: benchmark items derived from Stack Overflow are likely in these models' pretraining
data, so their scores are an UPPER BOUND — report them as such.
"""
import argparse
import json
import re
from pathlib import Path

from bench.harness import score_items
from cli.jqgen import build_context

BENCH = Path(__file__).resolve().parent / "nl2jq-bench.jsonl"

SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program (jq 1.7). Output ONLY the jq program on one line — no explanation, "
          "no markdown fences, no `jq` prefix.")


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    if text.startswith("jq "):
        text = text[3:].strip()
    return text.strip().strip("'").strip()


def anthropic_generate(model):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile

    def gen(item):
        context = build_context(item["input"], "auto")
        user = f"Request: {item['request']}\nJSON sample: {context}"
        # No temperature/top_p on current Claude models — they 400. Thinking omitted
        # (fast, cheap; jq generation doesn't need it). max_tokens small: programs are short.
        resp = client.messages.create(
            model=model, max_tokens=512, system=SYSTEM,
            messages=[{"role": "user", "content": user}])
        if resp.stop_reason == "refusal":
            return [""]
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return [_clean(text)]

    return gen


def openai_generate(model):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY

    def gen(item):
        context = build_context(item["input"], "auto")
        user = f"Request: {item['request']}\nJSON sample: {context}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}])
        return [_clean(resp.choices[0].message.content or "")]

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    model = a.model or ("claude-opus-4-8" if a.provider == "anthropic" else "gpt-5")
    items = [json.loads(l) for l in BENCH.open()]
    gen = anthropic_generate(model) if a.provider == "anthropic" else openai_generate(model)
    res = score_items(items, gen, k=1)
    summary = {k: round(v, 3) for k, v in res.items() if k != "details"}
    print(f"{a.provider} / {model} (zero-shot, UPPER BOUND — SO items may be in pretraining)")
    print(json.dumps(summary, indent=2))
    wrong = [d["id"] for d in res["details"] if not (d["results"] and d["results"][0]["correct"])]
    print(f"missed ({len(wrong)}):", wrong)


if __name__ == "__main__":
    main()
