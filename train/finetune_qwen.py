"""Fine-tune Qwen3-0.6B on the nl2jq data — the practical baseline (SPEC §5.2).

Runs on a rented CUDA GPU. Same (request, context) -> program mapping as the scratch
model, wrapped in Qwen's chat template so we inherit its pretrained jq/JSON priors.

    pip install transformers accelerate peft
    python -m train.finetune_qwen --data data/v2 --out artifacts/qwen06b --epochs 2
"""
import argparse
import json
from pathlib import Path

SYSTEM = ("You translate a natural-language request plus a sample of JSON into a single "
          "jq program. Output only the jq program, nothing else.")


def to_chat(row):
    user = f"Request: {row['request']}\nJSON sample: {row['context']}"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["program"]}]


def main():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="artifacts/qwen06b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="gradient accumulation steps; effective batch = bs * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #training rows (0=all); the 0.6B baseline needs only a subset")
    ap.add_argument("--lora", action="store_true", help="use LoRA instead of full FT")
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in (Path(a.data) / "train.jsonl").open()]
    if a.limit and a.limit < len(rows):
        # deterministic evenly-spaced subsample so all schema families stay represented
        step = len(rows) / a.limit
        rows = [rows[int(i * step)] for i in range(a.limit)]
    print(f"fine-tuning on {len(rows)} rows")

    def _ids(text):
        # tokenize=True can return a tokenizers.Encoding on newer transformers, which
        # pyarrow can't serialize — render to text first, then tokenize to a plain list.
        return list(tok(text, add_special_tokens=False)["input_ids"])

    def encode(row):
        ids = _ids(tok.apply_chat_template(to_chat(row), tokenize=False))
        # mask everything before the assistant turn so loss is on the program only
        prompt_ids = _ids(tok.apply_chat_template(to_chat(row)[:-1], tokenize=False,
                                                  add_generation_prompt=True))
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        ids, labels = ids[:a.max_len], labels[:a.max_len]
        return {"input_ids": ids, "labels": labels[:len(ids)],
                "attention_mask": [1] * len(ids)}

    # Plain torch Dataset — avoids datasets/pyarrow arrow serialization (which chokes on
    # this transformers/datasets version). Trainer + our collator handle the rest.
    class _DS(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    ds = _DS([encode(r) for r in rows])
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    # Gradient checkpointing trades compute for a large activation-memory saving; the
    # input_require_grads shim is needed so grads flow to LoRA adapters through the
    # checkpointed (frozen) base. Qwen3's ~152k vocab makes the loss-logits the memory
    # peak, so we also keep the micro-batch small and recover throughput via grad_accum.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, target_modules="all-linear",
            task_type="CAUSAL_LM"))

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [tok.pad_token_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=25, save_strategy="epoch", report_to=[],
        remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
