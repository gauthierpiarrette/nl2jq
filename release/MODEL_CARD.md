---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- jq
- code-generation
- json
- small-language-model
datasets:
- gauthierpiarrette/nl2jq
metrics:
- execution-accuracy
---

# nl2jq-40m

*Part of the **nl2jq** project — [all artifacts](https://huggingface.co/collections/gauthierpiarrette/nl2jq-natural-language-to-jq-locally-6a5299203df3832415275223) · [code + CLI](https://github.com/gauthierpiarrette/nl2jq) · [live demo](https://huggingface.co/spaces/gauthierpiarrette/nl2jq) · [benchmark](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench)*

A 37M-parameter decoder-only language model trained **from scratch** to translate a
natural-language request plus a JSON sample into a [jq](https://jqlang.github.io/jq/)
program. Runs locally on CPU in well under a second.

Research question: *how small can a model be and still write executable jq?*

## Usage

```bash
cat data.json | jqgen "total spend per customer, paid orders only"
```

Prompt format:
```
<|request|> {your request}
<|input|> {raw prefix or shape sketch of your JSON}
<|program|>
```
The model completes with the jq program followed by `<|end|>`.

## Results

Execution accuracy (output of the generated program equals the reference output) on the
frozen 400-item
[`nl2jq-bench v1.0.0`](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench)
(held-out by construction: 0% field overlap with training, novel domains, evaluated once
per system). **These weights are v7** — the third and final data generation of the
from-scratch experiment. The three-generation arc *is* the research result:

| system | frozen pass@1 | valid | what it taught us |
|---|---|---|---|
| v5 (scored 0.55 on the retired in-distribution dev split) | 0.00 | 0.48 | the dev score was **vocabulary recall**, not skill — on unseen fields it emits training-vocab names (`.urgent`) |
| v6 (per-example-unique field names) | *not run on the frozen set; ~0.01 on the dev twin* | — | name-level uniqueness isn't enough: BPE absorbed the name *components*, and real field names tokenize into fragments the model never learned to emit |
| **v7 (this: components from real-text subwords)** | **0.04** | 0.56 | token-level copying finally works — and 37M still cannot compose correct programs on OOD inputs |
| **v7 + input-grounded decoding** | **0.09** | 0.76 | the honest ceiling of a 37M from-scratch model with system help |
| [nl2jq-qwen3-0.6b](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b) (pretrained sibling, same data recipe) | 0.40 | 0.73 | what pretraining buys: 0.40 vs 0.09 |
| Claude Opus 4.8 (zero-shot, context row) | 0.96 | 0.98 | the task ceiling |

**Conclusion of the experiment:** execution-verified synthetic data teaches a 37M
from-scratch model jq *syntax* essentially completely (in-training execution accuracy
>0.9 on its own distribution), but out-of-distribution *semantics* — binding the user's
actual field names and composing the right operation — does not emerge at this scale,
even after redesigning the data twice against mechanistically-diagnosed failures. Treat
this model as the research artifact it is; for actual use, take the pretrained backends.

`pass@1` is greedy; `valid` = fraction of generated programs that parse and run under
jq 1.7.1. Scoring is execution equivalence (array/stream repackaging normalized;
array order normalized for items flagged `order_insensitive`). Frozen scores are one-shot
(no selection or iteration against the frozen split; see its
[FREEZE record](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench)).

- **The released checkpoint is the best checkpoint by dev-novel pass@1, not the final
  training step** — later checkpoints measurably overfit the synthetic distribution as the
  learning rate anneals. (A separate dev split was the selection signal; the frozen split
  was evaluated exactly once, after selection.)
- The frontier zero-shot row is *context, not a peer comparison*: the model is proprietary,
  far larger, and has surely seen public jq during pretraining. Its dev→frozen *rise*
  (0.75 → 0.96) indicates part of its dev-split misses were request-ambiguity artifacts,
  which the frozen benchmark's adversarial review eliminated.

### What to use this model for

Treat nl2jq-40m as the **research artifact** answering "how far does execution-verified
synthetic data get a from-scratch 37M model?" — the answer is: all the way on syntax,
in-distribution on semantics, and **barely** (0.04–0.09) on held-out generalization. For
actual CLI use on your own JSON, use the
[`nl2jq-qwen3-0.6b`](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b) backend.

## Architecture

Llama-shape (RMSNorm, RoPE, SwiGLU, tied embeddings), 10 layers, d_model 512, 8 heads,
2048 context, 12,288-token byte-level BPE vocab with single-digit number tokens and byte
fallback (the shipped v7 tokenizer; the retired v5 used a 10,490 vocab).
The weights load directly as a `transformers` `LlamaForCausalLM` — the conversion is a
pure key rename (same RoPE convention), verified to reproduce the original module's logits
**exactly** (max abs diff 0.0).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gauthierpiarrette/nl2jq-40m")
model = AutoModelForCausalLM.from_pretrained("gauthierpiarrette/nl2jq-40m")
```

> GGUF/llama.cpp note: the tokenizer splits numbers into single digits (deliberate, for
> numeric reasoning), a pre-tokenizer llama.cpp doesn't yet recognize, so a faithful GGUF
> export isn't currently possible — use the PyTorch/`transformers` path above.

## Training data

100% synthetic and execution-verified — see the
[nl2jq dataset](https://huggingface.co/datasets/gauthierpiarrette/nl2jq). No web text, no
scraped code. Every program was run against its input and kept only if it produced
non-degenerate output.

## Limitations

- **OOD composition is the failure mode — measured, not hypothetical.** v7 fixed
  field-copying at the token level, yet the model still reaches only 0.04 raw / 0.09
  grounded on the frozen benchmark: it can now emit your field names but usually wraps
  them in the wrong program. Research artifact, not a daily driver.
- jq 1.7.1 only; no streaming, modules, or user-defined functions.
- Best on the transform/filter/aggregate patterns people actually write; exotic programs
  are out of distribution.
- Ambiguous requests may yield a valid program that answers a different reading.
- **Inspect before trusting the output.** Generated jq should be reviewed before use — the
  `jqgen` CLI prints the program to stderr for this reason. jq runs locally with no network
  access, but an incorrect program can still produce misleading results.
- Not a general assistant — it emits jq and nothing else.

## License

Apache-2.0.

Code, benchmark tooling, and training pipeline: [github.com/gauthierpiarrette/nl2jq](https://github.com/gauthierpiarrette/nl2jq)
