---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-2B
tags:
- jq
- code-generation
- json
- lora
datasets:
- gauthierpiarrette/nl2jq
metrics:
- execution-accuracy
---

# nl2jq-qwen3.5-2b

*Part of the **nl2jq** project — [all artifacts](https://huggingface.co/collections/gauthierpiarrette/nl2jq-natural-language-to-jq-locally-6a5299203df3832415275223) · [code + CLI](https://github.com/gauthierpiarrette/nl2jq) · [live demo](https://huggingface.co/spaces/gauthierpiarrette/nl2jq) · [benchmark](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench)*

A LoRA fine-tune of [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) that translates a
natural-language request plus a JSON sample into a [jq](https://jqlang.org/) program.
**The most accurate local backend** in the nl2jq family — the `jqgen --backend qwen-2b`
option. The LoRA adapter is merged; it loads as an ordinary model.

The [0.6B sibling](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b) remains
the CLI's **default**; choose this backend when accuracy on everyday queries matters more
than latency (~3–8s per query on laptop CPU; note T5-generalization is slightly *lower*
than the 0.6B's — see the table).

## Results

Execution accuracy on the frozen 400-item
[`nl2jq-bench v1.0.0`](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench)
(0% field overlap with training, novel domains, one-shot evaluation — see the bench card):

| system | pass@1 | valid | T1 | T2 | T3 | T4 | T5-gen |
|---|---|---|---|---|---|---|---|
| this model, greedy | 0.46 | 0.75 | 0.67 | 0.61 | 0.47 | 0.33 | 0.13 |
| **this model + exec-rerank k=4 (the CLI config)** | **0.48** | **0.82** | **0.75** | 0.61 | 0.47 | 0.35 | 0.18 |
| nl2jq-qwen3-0.6b-v6 (smaller sibling) | 0.40 | 0.73 | 0.65 | 0.52 | 0.36 | 0.28 | 0.20 |
| Claude Opus 4.8 (zero-shot, context row) | 0.96 | 0.98 | 1.00 | 0.96 | 0.96 | 0.96 | 0.90 |

Read this as: **reliable on elementary and core tasks over never-seen field names**
(T1 0.75 / T2 0.61 in the CLI configuration), progressively weaker on multi-stage
compositions and exotic constructs. The frontier row is context, not a peer comparison.
The exec-rerank config samples 4 candidates, repairs field references against the keys
actually present in your JSON, and returns the first candidate that executes informatively
— it is what `jqgen` runs by default.

## Usage

```bash
cat data.json | jqgen --backend qwen-2b "total ridership per garage, highest first"
```

Or directly (non-thinking chat model — no reasoning-block handling needed):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gauthierpiarrette/nl2jq-qwen3.5-2b")
model = AutoModelForCausalLM.from_pretrained("gauthierpiarrette/nl2jq-qwen3.5-2b")
msgs = [{"role": "system", "content": "You translate a natural-language request plus a "
         "sample of JSON into a single jq program. Output only the jq program, nothing else."},
        {"role": "user", "content": 'Request: highest fare\nJSON sample: [{"rider":"a","fare":12}]'}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                              enable_thinking=False)
ids = enc["input_ids"]
out = model.generate(ids, max_new_tokens=128,
                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
```

## Training

LoRA (r=32, α=64, all-linear), 1 epoch over 150k execution-verified rows of the v7
synthetic data generation (per-example-unique field names built from real-text subword
components, so copying — not recall — is the only strategy that works; plus generators for
`reduce`/`foreach`/`walk`/`paths`/`if` and friends). All data is synthetic and
execution-verified under jq 1.7.1; no scraped content; no frontier-model outputs.
Development selection used a separate dev split — the frozen benchmark was evaluated
exactly once.

## Limitations

- Multi-stage compositions (T4) and exotic constructs (T5) remain weak (0.35 / 0.18) —
  inspect the program before trusting output (the CLI prints it to stderr and supports
  `--no-run`).
- ~2–8s per query on laptop CPU via transformers; faster on any GPU / Apple Silicon.
- jq 1.7.1 semantics; no modules/streaming.

## License

Apache-2.0 (base model Apache-2.0; fine-tuned on CC BY 4.0 synthetic data).

## GGUF / llama.cpp status

Not shipped, deliberately. A Q4_K_M conversion works mechanically (note: the base is a
vision-language wrapper — a peft-merged text model must be re-wrapped with the base's
vision tensors before `convert_hf_to_gguf.py` produces a loadable file), and the result
loads without errors — but in our testing (llama.cpp b9957/b9964, CPU inference on macOS
and Linux) generation for this hybrid GDN+MoE architecture never completed in reasonable
time. We don't publish artifacts whose output we could not verify. Use the `transformers`
path above; a GGUF will follow when llama.cpp's kernels for this architecture mature.

Code, benchmark tooling, and training pipeline: [github.com/gauthierpiarrette/nl2jq](https://github.com/gauthierpiarrette/nl2jq)
