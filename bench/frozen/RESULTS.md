# nl2jq-bench frozen results (append-only)

One row per official run. No checkpoint selection against this file.

| date | model | pass@1 | valid@1 | T1 | T2 | T3 | T4 | T5 | T5-gen | bench sha |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-10 | nl2jq-40m (37M scratch) | 0.00 | 0.48 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | `25e7c0136070` |
| 2026-07-10 | Claude Opus 4.8 (zero-shot) | 0.96 | 0.98 | 1.00 | 0.96 | 0.96 | 0.96 | 0.90 | 0.90 | `25e7c0136070` |
| 2026-07-10 | nl2jq-qwen3-0.6b (LoRA) | 0.24 | 0.67 | 0.45 | 0.38 | 0.21 | 0.12 | 0.02 | 0.02 | `25e7c0136070` |
| 2026-07-10 | Qwen3-0.6B base (zero-shot, thinking) | 0.01 | 0.06 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | `25e7c0136070` |
| 2026-07-10 | nl2jq-qwen3-0.6b-v6 (LoRA, v6 data) | 0.40 | 0.73 | 0.65 | 0.52 | 0.36 | 0.28 | 0.20 | 0.20 | `25e7c0136070` |
| 2026-07-10 | nl2jq-40m-v7 (37M scratch, subword-fixed data) | 0.04 | 0.56 | 0.08 | 0.08 | 0.02 | 0.01 | 0.02 | 0.02 | `25e7c0136070` |
| 2026-07-10 | nl2jq-40m-v7 + input-grounded decoding | 0.09 | 0.76 | 0.17 | 0.14 | 0.08 | 0.01 | 0.05 | 0.05 | `25e7c0136070` |
| 2026-07-10 | nl2jq-qwen3.5-2b-v7 (LoRA, v7 data) | 0.46 | 0.75 | 0.67 | 0.61 | 0.47 | 0.33 | 0.13 | 0.13 | `25e7c0136070` |
| 2026-07-11 | nl2jq-qwen3.5-2b-v7 + exec-rerank k=4 (CLI config) | 0.48 | 0.82 | 0.75 | 0.61 | 0.47 | 0.35 | 0.18 | 0.18 | `25e7c0136070` |

Notes:
- "Qwen3-0.6B base (zero-shot, thinking)": greedy, 1024-token budget incl. thinking; 56/400
  outputs were truncated mid-reasoning and score invalid under these conditions. Sampled
  non-truncated outputs show the dominant failure is genuine (pseudo-code / shell-wrapped /
  malformed jq), so the true zero-shot range is ~0.01-0.05, not materially higher.
- "+ input-grounded decoding" and "+ exec-rerank" rows are SYSTEM configurations (the same
  model with k-candidate sampling, field repair against the input's keys, and execution
  filtering — cli/decoding.py). They are what the jqgen CLI ships; raw rows are the model
  alone.
- Development selection (checkpoints, recipes, decoding parameters) used the dev-novel
  split only; every row here is the first and only evaluation of that system on this file.
