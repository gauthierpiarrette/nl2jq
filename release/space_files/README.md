---
title: nl2jq — English to jq
emoji: 🧭
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
models:
- gauthierpiarrette/nl2jq-qwen3-0.6b
---

# nl2jq — English → jq

Paste JSON, ask in plain English, get a jq program **and** its output. Runs the
0.6B fast backend of the [nl2jq project](https://huggingface.co/datasets/gauthierpiarrette/nl2jq-bench)
entirely inside this Space — your JSON goes nowhere else.

Honest expectations: frozen-benchmark pass@1 is **0.40** for this backend (see the
[model card](https://huggingface.co/gauthierpiarrette/nl2jq-qwen3-0.6b)); free-CPU
queries take ~15–60s. The generated program is always displayed — read it before
trusting the result. For real use, install the CLI: `pip install -e .` from the [repo](https://github.com/gauthierpiarrette/nl2jq).
