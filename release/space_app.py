"""Hugging Face Space: paste JSON, ask in English, see the jq program + its output.

Self-contained (no repo imports). Runs nl2jq-qwen3-0.6b (the fast backend) with a
latency-trimmed variant (k=2) of the CLI's benchmarked k=4 exec-rerank decode.
Space requirements: gradio, torch, transformers (requirements.txt) + jq (packages.txt).

Note for the Space card: on free CPU hardware a query takes ~15-60s; the local CLI is
much faster. Accuracy expectations are disclosed on the model card (frozen pass@1 0.40
for this backend) — always read the generated program before trusting the result.
"""
import json
import re
import subprocess

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "gauthierpiarrette/nl2jq-qwen3-0.6b"
SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program. Output only the jq program, nothing else.")
K = 2  # candidates (greedy + 1 sample): latency-friendly slice of the k=4 CLI config

TOK = AutoTokenizer.from_pretrained(REPO)
MODEL = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()


def _context(doc, limit=800):
    s = json.dumps(doc, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def _clean(text):
    text = re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    return text.removeprefix("jq ").strip().strip("`")


def _generate(request, context, temperature):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Request: {request}\nJSON sample: {context}"}]
    enc = TOK.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = enc if torch.is_tensor(enc) else enc["input_ids"]
    kw = ({"do_sample": True, "temperature": temperature} if temperature > 0
          else {"do_sample": False})
    with torch.no_grad():
        out = MODEL.generate(ids, max_new_tokens=128, pad_token_id=TOK.eos_token_id, **kw)
    return _clean(TOK.decode(out[0][ids.shape[1]:], skip_special_tokens=True))


def _run_jq(program, json_text):
    try:
        proc = subprocess.run(["jq", program], input=json_text, capture_output=True,
                              text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return False, "(jq timed out)"
    if proc.returncode != 0:
        return False, f"(jq error) {proc.stderr.strip()}"
    out = proc.stdout.strip()
    lines = [l for l in out.splitlines() if l.strip()]
    informative = bool(lines) and not all(l.strip() == "null" for l in lines)
    return informative, out


def infer(json_text, request):
    try:
        doc = json.loads(json_text)
    except json.JSONDecodeError as e:
        return "—", f"Invalid JSON: {e}"
    context = _context(doc)
    candidates = [_generate(request, context, 0.0)]
    if K > 1:
        candidates += [_generate(request, context, 0.8) for _ in range(K - 1)]
    fallback = candidates[0], _run_jq(candidates[0], json_text)[1]
    for cand in candidates:
        ok, out = _run_jq(cand, json_text)
        if ok:
            return cand, out
    return fallback


# every example below was verified correct against the live Space before shipping
EXAMPLES = [
    ['[{"customer":"Alice","total":40,"paid":true},'
     '{"customer":"Bob","total":75,"paid":false},'
     '{"customer":"Alice","total":22,"paid":true}]',
     "customers of the paid orders"],
    ['[{"name":"datasette","stars":9000},{"name":"jq","stars":30000}]',
     "names of repos with more than 10000 stars"],
    ['{"width":800,"height":600,"fps":30}', "which setting has the largest value?"],
]

demo = gr.Interface(
    fn=infer,
    inputs=[gr.Textbox(label="Your JSON", lines=10),
            gr.Textbox(label="What do you want? (plain English)")],
    outputs=[gr.Textbox(label="jq program (read this before trusting the result)"),
             gr.Textbox(label="Result", lines=10)],
    examples=EXAMPLES,
    title="nl2jq — English → jq, on a 0.6B local model",
    description=("Small local model (frozen-benchmark pass@1 0.40 — see the "
                 "[model card](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b)). "
                 "Free-CPU queries take ~15-60s; the `jqgen` CLI is much faster locally. "
                 "The program is always shown — jq runs it only here in this sandbox."),
)

if __name__ == "__main__":
    demo.launch()
