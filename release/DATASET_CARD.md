---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
tags:
- jq
- json
- code-generation
- synthetic
size_categories:
- 1M<n<10M
---

# nl2jq

A fully synthetic, execution-verified dataset for translating natural language into
[jq](https://jqlang.github.io/jq/) programs. The first public dataset for this task.

## Each row

```json
{
  "request": "total spend per customer, paid orders only",
  "context": "[{\"id\": 1, \"customer\": {\"name\": \"Alice\"}, ...}]",
  "context_mode": "raw",
  "program": "map(select(.status==\"paid\")) | group_by(.customer.name) | ...",
  "expected_output": [[{"customer": "Alice", "total": 62}]],
  "input_doc": {"...": "the JSON the program was verified against"},
  "tags": ["filter", "group_by", "sum"],
  "tier": 2,
  "domain": "commerce",
  "split": "train"
}
```

- `context` is what a model should condition on: either a raw prefix of the JSON
  (`context_mode: raw`) or a compact shape sketch for large inputs (`shape`).
- `program` is guaranteed to run under jq 1.7.1 and produce `expected_output` on
  `input_doc`.

## Construction

Schema → documents → type-directed jq program → **execute** → filter degenerate/erroring
programs → behavioral dedup → natural-language description (templates + paraphrases).
Validation holds out entire schema families and program templates, so models are tested
on compositional generalization rather than memorization. Full method in the
[project spec](https://github.com/gauthierpiarrette/nl2jq/blob/main/SPEC.md); generation code in the
[project repo](https://github.com/gauthierpiarrette/nl2jq).

## Statistics

| | |
|---|---|
| Train rows | 1,905,710 |
| Validation rows | 75,981 |
| Schema families | ~1.27M distinct; 63,105 held out for validation, **0 train/val overlap** |
| Field vocabulary | 500+ field names incl. compound names (`unit_price`, `daily_stock`) so models must copy field names from the input rather than memorize a fixed set |
| Top-level shapes | arrays of records, single objects, homogeneous number/bool objects, scalar/string/nested arrays, key-value pairs |
| Context mode | ~60% raw prefix / ~40% shape sketch on train, ~50/50 on val (raw is forced for primitive shapes) |
| Verification | 100% executed under jq 1.7.1; degenerate/erroring/constant-output programs dropped |

Every program is behaviorally deduplicated (programs with identical outputs across the
sampled documents are collapsed), and validation holds out **entire schema families**, so
a model cannot score on val by memorizing a family seen in train.

## Verifying a row yourself

```bash
jq -c '<program>' <<< '<input_doc>'   # equals expected_output
```

## License

CC BY 4.0. Contains no scraped or copyrighted content — schemas, values, and programs are
all generated. Field-name vocabularies are drawn from public API shapes (structure only).

Code: [github.com/gauthierpiarrette/nl2jq](https://github.com/gauthierpiarrette/nl2jq)
