---
license: cc-by-4.0
task_categories:
- text-generation
tags:
- jq
- code-generation
- json
- execution-benchmark
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: nl2jq-bench-1.0.0.jsonl
- config_name: devset-v0
  data_files:
  - split: dev
    path: devset-v0.jsonl
---

# nl2jq-bench

An **execution-scored, frozen** benchmark for natural-language → [jq](https://jqlang.org/)
program generation. Each item is a realistic request over a small JSON input with a
verified reference program; models are scored on whether their program's **output** matches
the reference output under jq 1.7.1 — not on string similarity.

## Files

| file | what |
|---|---|
| `nl2jq-bench-1.0.0.jsonl` | **the benchmark**: 400 frozen items (v1.0.0) |
| `FREEZE.txt` | the freeze record — content sha256 (the benchmark's identity), tier histogram, jq binary hash |
| `harness.py` | standalone scorer (Python 3.10+, any jq binary; pin jq 1.7.1 to reproduce) |
| `devset-v0.jsonl` | the RETIRED 100-item development split — see "History" below; not a benchmark |

Cite scores as `nl2jq-bench@1.0.0` together with the sha256 in `FREEZE.txt`. Any change to
the items is a new version with a new hash; v1.0.0 never changes.

## Design

**400 items, five tiers** (each item carries `tier` = `difficulty` 1–5):

| tier | n | content |
|---|---|---|
| T1 Elementary | 60 | single op: path/index/slice/`length`/`keys`/`has`/`type`/`//` |
| T2 Core | 100 | `select`/`map`, comparisons, arithmetic, `unique`/`sort`/`min`/`max`/`add`, `contains` |
| T3 Aggregation & reshape | 100 | `group_by`/`sort_by`/`*_by`, `to_entries` family, `map_values`, `flatten`, `any`/`all`, object construction |
| T4 Strings, formats & composition | 80 | `split`/`join`/`test`/`capture`/`sub`/`gsub`, casts, interpolation, `@csv`/`@tsv`/`@json`/`@base64`, 3+-stage pipelines |
| T5 **Generalization** | 60 | `reduce`, `foreach`, `walk`/`..`, `paths`/`getpath`, `try/catch`, `if/then/elif`, object-merge `add`, computed-key `group_by`, `INDEX` |

**Anti-contamination, by construction** (relative to the
[`nl2jq`](https://huggingface.co/datasets/gauthierpiarrette/nl2jq) training set):

- **Field-name disjointness: 0% overlap.** Every input field name was checked against the
  full training vocabulary (687 names incl. constructible compounds) with a near-miss bar
  (no shared ≥4-char prefixes, no small-edit variants, no common abbreviations) and an
  enum-value gate (no training enum value appears as an input value or program literal).
  The gate ships in the project repo ([`bench/validate_novelty.py`](https://github.com/gauthierpiarrette/nl2jq/blob/main/bench/validate_novelty.py)) and rebuilds the
  forbidden vocabulary from the generator source at check time.
- **T5 is beyond-grammar.** The 60 T5 items use only constructs the training-data grammar
  provably never emits (verified by an emission audit + a source gate,
  [`bench/audit_coverage.py`](https://github.com/gauthierpiarrette/nl2jq/blob/main/bench/audit_coverage.py)). **Report the T5-only score separately** — it is the
  generalization headline; a model cannot have seen these operation shapes in `nl2jq`.
- **Novel domains.** 12 domains disjoint from the training set's (clinical, gradebook,
  transit, sports, music, weather, real-estate, recipes, library, gaming, lab, civic),
  plus ≤2% abstract single-letter-key items (tagged `abstract: true`).
- **Sealed canary.** A 100-item sealed set was frozen alongside v1.0.0 (sha256 in
  `FREEZE.txt`, never published). It exists to detect future train-on-test: a large
  public−sealed gap for a model is a contamination signal. Every item also carries a
  unique `canary_id` for corpus grepping.

**Review process.** Every item was execution-verified under a pinned jq 1.7.1 binary
(expected outputs are pinned from execution, never typed), then adversarially reviewed for
ambiguity: where a *different but defensible* program produces a different output, the
request was disambiguated or the alternative's **executed** output added to
`acceptable_outputs`. 11% of items were revised in this pass.

## Item format

```json
{
  "id": "v1-042", "tier": 3, "difficulty": 3, "domain": "transit",
  "request": "total ridership per garage",
  "input": [ {"route_no": "41B", "garage": "Elmside", "ridership": 10432}, ... ],
  "reference_program": "group_by(.garage) | map({garage: .[0].garage, ridership: (map(.ridership) | add)})",
  "expected_output": [ ... ],
  "order_insensitive": false, "abstract": false,
  "novel_fields": ["route_no", "garage", "ridership"],
  "grammar_covered": true, "provenance": "hand", "source": "original",
  "jq_version": "1.7.1", "frozen_in": "1.0.0", "canary_id": "…", "license": "CC-BY-4.0"
}
```

Some items carry `acceptable_outputs` (a list of equally-correct output streams).

## Scoring

A prediction is **correct** if its jq output equals an acceptable output under these
normalizations (see `harness.py`):

- **array/stream equivalence** — `map(.x)` and `.[].x` score the same;
- **array-order-insensitivity** for items flagged `order_insensitive`;
- **float tolerance** for numeric outputs.

`harness.py` is standalone — it needs only Python 3.10+ and a jq binary (uses `$JQ_BIN`,
else `jq` on PATH; pin jq 1.7.1 to reproduce published numbers exactly):

```python
from harness import score_program
score_program('map(.route_no)', item)   # -> {"valid": True, "correct": ...}
```

**One-shot discipline:** this benchmark is meant to be evaluated once per model release.
Do not select checkpoints, tune prompts, or iterate data against it — that is what
`devset-v0.jsonl` is for.

## Results (v1.0.0)

The complete run-by-run record lives in this repo's **`RESULTS.md`** (append-only; one row
per official run). Summary as of 2026-07-11 — rows marked *(superseded)* are earlier
fine-tunes kept for the record; the shipped models are the bold rows:

| system | pass@1 | valid@1 | T1 | T2 | T3 | T4 | **T5-gen** |
|---|---|---|---|---|---|---|---|
| 40m v5, from scratch *(superseded)* | 0.00 | 0.48 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **nl2jq-40m (v7 weights)** | 0.04 | 0.56 | 0.08 | 0.08 | 0.02 | 0.01 | 0.02 |
| nl2jq-40m + input-grounded decoding | 0.09 | 0.76 | 0.17 | 0.14 | 0.08 | 0.01 | 0.05 |
| Qwen3-0.6B base, zero-shot | 0.01 | 0.06 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 |
| 0.6B v5 LoRA *(superseded)* | 0.24 | 0.67 | 0.45 | 0.38 | 0.21 | 0.12 | 0.02 |
| **nl2jq-qwen3-0.6b (v6 weights)** | 0.40 | 0.73 | 0.65 | 0.52 | 0.36 | 0.28 | 0.20 |
| **nl2jq-qwen3.5-2b (v7 LoRA)** | 0.46 | 0.75 | 0.67 | 0.61 | 0.47 | 0.33 | 0.13 |
| nl2jq-qwen3.5-2b + exec-rerank k=4 (CLI config) | 0.48 | 0.82 | 0.75 | 0.61 | 0.47 | 0.35 | 0.18 |
| Claude Opus 4.8 (zero-shot) | 0.96 | 0.98 | 1.00 | 0.96 | 0.96 | 0.96 | 0.90 |

For contrast, on the retired in-distribution dev split the earlier models scored 0.55 /
0.81 / 0.75 pass@1. The differences **are the measured generalization gap**, and
quantifying it honestly is the point of this benchmark. Three regimes emerge:

- **From scratch at 37M**: in-distribution skill barely transfers (dev 0.55 → frozen
  0.00–0.09 across three data generations). Syntax survives the distribution shift;
  field binding and composition do not.
- **Pretrained bases + task LoRA**: fine-tuning does all the work (base 0.01 → 0.40 at
  0.6B; 0.48 at 2B in the CLI configuration) — strong on elementary tasks over never-seen
  fields, progressively weaker on multi-stage composition and the T5 constructs.
- **Frontier zero-shot** climbs from 0.75 to 0.96: a share of its dev-split misses were
  ambiguity artifacts that this benchmark's adversarial review eliminated
  (disambiguated requests, executed `acceptable_outputs`).

## History: devset-v0

The original 100-item set (shipped here as `devset-v0.jsonl`) was used during development
for checkpoint selection and grammar iteration, and its field names overlap the training
vocabulary (~65%) — a post-hoc audit found 95/100 of its items structurally reproducible
by the training-data generator. It is retired as a benchmark and kept only for
reproducibility of the development history. **Do not report devset numbers as held-out
results.**

## Provenance

All items are original, authored for this benchmark (no scraped content, no Stack
Overflow), execution-verified under jq 1.7.1. Reference programs and inputs are
CC BY 4.0.

Code: [github.com/gauthierpiarrette/nl2jq](https://github.com/gauthierpiarrette/nl2jq).
Companion training set: [`gauthierpiarrette/nl2jq`](https://huggingface.co/datasets/gauthierpiarrette/nl2jq).
Models: [`nl2jq-40m`](https://huggingface.co/gauthierpiarrette/nl2jq-40m),
[`nl2jq-qwen3-0.6b`](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b),
[`nl2jq-qwen3.5-2b`](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3.5-2b).
