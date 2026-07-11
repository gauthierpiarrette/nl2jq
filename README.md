# nl2jq

Natural language → [jq](https://jqlang.org/), running locally. A synthetic-data pipeline,
a **frozen execution benchmark**, three models, and a CLI — every number in this project
is execution-verified, and the benchmark was built *after* auditing our own dev results
for contamination.

```bash
cat orders.json | jqgen "total spend per customer, paid orders only"
```
```
jq program: group_by(.customer) | map({customer: .[0].customer, total: (map(.total)|add)})
[{"customer":"Alice","total":62},{"customer":"Bo","total":41}]
```

Local, offline, private. The generated program is always printed before anything runs.

## The artifacts

| Artifact | What | Where |
|---|---|---|
| `nl2jq` dataset | 1.9M `(JSON, request, program, verified output)` rows, fully synthetic, execution-verified | [HF dataset](https://huggingface.co/datasets/gauthierpiarrette/nl2jq) |
| `nl2jq-bench` **v1.0.0** | Frozen 400-item execution benchmark: sha256-identified, 0% field overlap with training, a beyond-grammar generalization tier, a sealed canary set, an append-only results ledger | [HF dataset](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench) |
| `nl2jq-qwen3.5-2b` | The **accurate** local backend (Qwen3.5-2B LoRA) | [HF model](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3.5-2b) |
| `nl2jq-qwen3-0.6b` | The **fast** local backend (~1–3s/query CPU) | [HF model](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b) |
| `nl2jq-40m` | The **research artifact**: a 37M model trained from scratch — *how far does execution-verified synthetic data get you without pretraining?* (Answer: all the way on syntax, nowhere on OOD semantics — see the model card) | [HF model](https://huggingface.co/gauthierpiarrette/nl2jq-40m) |
| `jqgen` | The CLI (this repo) | `pip install -e .` |

## Results — frozen benchmark, every row one-shot

| system | pass@1 | valid | T1 elem. | T5 generalization |
|---|---|---|---|---|
| nl2jq-40m (37M from scratch) | 0.04 | 0.56 | 0.08 | 0.02 |
| nl2jq-40m + input-grounded decoding | 0.09 | 0.76 | 0.17 | 0.05 |
| nl2jq-qwen3-0.6b | 0.40 | 0.73 | 0.65 | 0.20 |
| nl2jq-qwen3.5-2b | 0.46 | 0.75 | 0.67 | 0.13 |
| **nl2jq-qwen3.5-2b + exec-rerank k=4 (the CLI default)** | **0.48** | **0.82** | **0.75** | 0.18 |
| Claude Opus 4.8, zero-shot (context row) | 0.96 | 0.98 | 1.00 | 0.90 |

Full per-tier table and run conditions: the bench repo's `RESULTS.md`. The benchmark is
deliberately hard: field names share **zero** vocabulary with the training data, so
models must read *your* JSON rather than recall theirs. Details and the freeze protocol
are in the [bench card](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench);
our earlier (higher) dev-split numbers were retired after a contamination audit — that
story is part of the project.

## Using the CLI

```bash
pip install -e .
cat data.json | jqgen "which items are out of stock?"          # fast 0.6B backend
cat data.json | jqgen --backend qwen-2b "top 3 dept by spend"  # accurate 2B backend
jqgen --no-run "..."   # print the program, don't execute
jqgen --k 1 "..."      # single greedy generation (default: 4 candidates + execution filter)
```

Which backend?

| backend | frozen pass@1 | CPU latency | pick when |
|---|---|---|---|
| `qwen` (0.6B, **default**) | 0.40 | ~1–3s | everyday use |
| `qwen-2b` | 0.48 (CLI config) | ~3–8s | accuracy over speed (T5 slightly lower) |

The default decode is the benchmarked configuration: sample 4 candidates, repair field
references against the keys actually present in your JSON, return the first candidate
whose execution produces informative output. Models auto-download from the Hub on first
use; nothing leaves your machine at inference time.

**Always inspect the generated program** (printed to stderr) before trusting the output —
these are small local models with a disclosed error profile, not oracles.

## How the data works

Every training example is machine-verified: sample a JSON schema → sample documents →
sample a type-directed jq program → **execute it** → keep only programs that run and
produce non-degenerate output → dedup by behavior → attach an author-written natural-
language description. v7 additionally synthesizes field names from a 34,799-word
real-text vocabulary so that *copying from the input* — not vocabulary recall — is the
only strategy that reduces loss (the three-generation story of why is on the
[`nl2jq-40m` model card](https://huggingface.co/gauthierpiarrette/nl2jq-40m)).

Because correctness is `execute(program, input) == expected`, the same signal drives
data filtering, benchmark scoring, candidate reranking, and any future RL.

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt          # add -r requirements-train.txt on a GPU box
# place a jq 1.7.1 binary at bin/jq (or `export JQ_BIN=$(which jq)`)

# 1. data (v7 recipe; ~25 min on a many-core box, hours on a laptop)
python -m pipeline.build_parallel --n 2000000 --out data/v7 --workers 12

# 2. tokenizer + from-scratch model (GPU, ~4h on one RTX 5090)
# (--vocab is a ceiling; BPE may saturate below it — the shipped model landed at 10,490)
python -m train.tokenizer --data data/v7 --vocab 12288 --out artifacts/tok
python -m train.train --config 40m --data data/v7 --tok artifacts/tok \
    --steps 50000 --batch 64 --keep_ckpts 1 --out artifacts/nl2jq-40m

# 3. LoRA a pretrained base (GPU, ~1-5h depending on base size)
python -m train.finetune_qwen --data data/v7 --model Qwen/Qwen3.5-2B \
    --out artifacts/qwen2b --limit 150000 --lora --lora_r 32 --lora_alpha 64 \
    --epochs 1 --bs 8 --grad_accum 2 --lr 1e-4 --max_len 768
python -m train.merge_qwen --base Qwen/Qwen3.5-2B --adapter artifacts/qwen2b \
    --out artifacts/qwen2b-merged

# 4. select checkpoints on a DEV split, then evaluate the frozen bench ONCE
python -m bench.eval_ckpts --dir artifacts/nl2jq-40m --tok artifacts/tok \
    --items bench/devnovel/devnovel-v1.jsonl
python -m bench.eval_frozen --backend qwen --model artifacts/qwen2b-merged \
    --label "my-run"

# 5. use it
cat data.json | python -m cli.jqgen --model artifacts/qwen2b-merged --backend qwen-2b "..."
```

**Benchmark discipline** (enforced by the tooling): checkpoint and recipe selection use
dev splits only; `bench/eval_ckpts.py` refuses frozen paths; the frozen set is evaluated
once per system and appended to the ledger.

## Layout

`pipeline/` synthetic data (schemas → docs → grammar → execute) · `train/` tokenizer +
from-scratch model + LoRA · `bench/` frozen benchmark, gates, harness, one-shot runner ·
`cli/` jqgen + grounded decoding · `grammars/jq.gbnf`.

## License

Code Apache-2.0. Dataset CC BY 4.0. Benchmark CC BY 4.0 (original items, no scraped
content). Models Apache-2.0. No frontier-model outputs anywhere in the training data;
frontier APIs were used only to measure baseline rows.
