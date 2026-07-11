"""jqgen — pipe JSON, ask in English, get (and run) a jq program.

    cat data.json | jqgen "total spend per customer, paid orders only"
    curl -s api/issues | jqgen "count by user.login, top 3" --no-run

Local, offline, CPU. The model is downloaded from the Hugging Face Hub on first use
and cached; nothing leaves your machine at inference time. The generated program is
printed to stderr so you can inspect it before trusting the result on stdout.

Backends:
  qwen (default)   Qwen3-0.6B fine-tune — fast (~1-3s CPU), frozen-bench pass@1 0.40.
  qwen-2b          Qwen3.5-2B fine-tune — most accurate local option (0.48 in the
                   default k=4 exec-rerank configuration).
  flagship         the 37M from-scratch research model (see its card before using).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.common import format_example  # noqa: E402
from pipeline.shape import raw_prefix, shape_sketch  # noqa: E402

FLAGSHIP_REPO = "gauthierpiarrette/nl2jq-40m"
QWEN_REPO = "gauthierpiarrette/nl2jq-qwen3-0.6b"
QWEN2B_REPO = "gauthierpiarrette/nl2jq-qwen3.5-2b"
QWEN_SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
               "jq program. Output only the jq program, nothing else.")


def resolve_jq() -> str:
    """Prefer a system jq; fall back to the copy bundled in the repo."""
    found = shutil.which("jq")
    if found:
        return found
    bundled = ROOT / "bin" / "jq"
    if bundled.exists():
        return str(bundled)
    sys.exit("jqgen: jq not found. Install it (brew install jq / apt install jq) and retry.")


def build_context(doc, mode):
    if mode == "shape":
        return shape_sketch(doc)
    if mode == "raw":
        return raw_prefix(doc)
    # auto: shape if the raw prefix would truncate badly
    raw = raw_prefix(doc)
    return shape_sketch(doc) if raw.endswith("…") else raw


def _clean_program(text: str) -> str:
    # Qwen3 is a reasoning model and emits a <think>…</think> block first — drop it.
    text = re.sub(r"(?s)^\s*<think>.*?</think>\s*", "", text)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    if text.startswith("jq "):
        text = text[3:]
    return text.strip().strip("`").strip()


def load_model(repo_or_path, device):
    """Load any Hub repo id or local dir via transformers (auto-downloads + caches)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo_or_path)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    # torch_dtype is accepted across transformers 4.x and 5.x (dtype= is 5.x-only)
    model = AutoModelForCausalLM.from_pretrained(repo_or_path, torch_dtype=dtype).to(device).eval()
    return model, tok


def generate_flagship(model, tok, request, context, device, max_new=128, temperature=0.0):
    prompt = format_example(request, context, None)  # <|request|> … <|program|>
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    eos = tok.convert_tokens_to_ids("<|end|>")
    kw = ({"do_sample": True, "temperature": temperature} if temperature > 0
          else {"do_sample": False})
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, eos_token_id=eos,
                             pad_token_id=0, **kw)
    cont = out[0][ids.shape[1]:].tolist()
    if eos in cont:
        cont = cont[: cont.index(eos)]
    return _clean_program(tok.decode(cont, skip_special_tokens=True))


def generate_qwen(model, tok, request, context, device, max_new=128, temperature=0.0):
    msgs = [{"role": "system", "content": QWEN_SYSTEM},
            {"role": "user", "content": f"Request: {request}\nJSON sample: {context}"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    # newer transformers return a BatchEncoding here, not a bare tensor
    ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
    kw = ({"do_sample": True, "temperature": temperature} if temperature > 0
          else {"do_sample": False})
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new,
                             pad_token_id=tok.eos_token_id, **kw)
    return _clean_program(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser(description="Natural language -> jq (local, execution-checked)")
    ap.add_argument("request", help="what you want, in plain English")
    ap.add_argument("--backend", choices=["qwen", "qwen-2b", "flagship"], default="qwen",
                    help="qwen = Qwen3-0.6B fine-tune, fast (default); "
                         "qwen-2b = Qwen3.5-2B fine-tune, most accurate local option; "
                         "flagship = the 37M from-scratch research model")
    ap.add_argument("--model", default=None,
                    help="override the model (Hub repo id or local dir); "
                         "defaults to the backend's published repo")
    ap.add_argument("-m", "--mode", choices=["auto", "raw", "shape"], default="auto")
    ap.add_argument("--k", type=int, default=4,
                    help="candidates to sample + execution-filter (1 = single greedy). "
                         "The default k=4 exec-rerank is the benchmarked CLI config.")
    ap.add_argument("--no-run", action="store_true", help="print the program, don't execute")
    ap.add_argument("--quiet", action="store_true", help="only print jq's output")
    args = ap.parse_args()

    raw_in = sys.stdin.read()
    try:
        doc = json.loads(raw_in)
    except json.JSONDecodeError as e:
        print(f"jqgen: stdin is not valid JSON ({e})", file=sys.stderr)
        sys.exit(2)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    context = build_context(doc, args.mode)
    repo = args.model or {"qwen": QWEN_REPO, "qwen-2b": QWEN2B_REPO,
                          "flagship": FLAGSHIP_REPO}[args.backend]
    model, tok = load_model(repo, device)
    gen = generate_flagship if args.backend == "flagship" else generate_qwen
    if args.k <= 1:
        program = gen(model, tok, args.request, context, device)
    else:
        # the benchmarked CLI config: greedy + sampled candidates, field repair against
        # the input's actual keys, first informative execution wins (cli/decoding.py)
        from cli.decoding import grounded_pick
        cands = [gen(model, tok, args.request, context, device)]
        cands += [gen(model, tok, args.request, context, device, temperature=0.8)
                  for _ in range(args.k - 1)]
        program, _meta = grounded_pick(cands, doc, args.request, resolve_jq())

    if not args.quiet:
        print(f"\033[2mjq program:\033[0m {program}", file=sys.stderr)
    if args.no_run:
        print(program)
        return
    proc = subprocess.run([resolve_jq(), program], input=raw_in, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"jqgen: jq failed: {proc.stderr.strip()}", file=sys.stderr)
        print(program)
        sys.exit(1)
    sys.stdout.write(proc.stdout)


if __name__ == "__main__":
    main()
