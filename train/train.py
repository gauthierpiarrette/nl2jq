"""Train an nl2jq model (sanity or 40m) with in-loop execution-accuracy eval.

    python -m train.train --config sanity --data data/v0 --tok artifacts/tok \
        --steps 3000 --out artifacts/sanity
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import JQDataset
from .eval_exec import eval_exec
from .model import CONFIGS, NL2JQModel
from .tokenizer import load_tokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step, warmup, total, peak, floor_frac=0.1):
    if step < warmup:
        return peak * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sanity")
    ap.add_argument("--data", default="data/v0")
    ap.add_argument("--tok", default="artifacts/tok")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--ctx_loss", type=float, default=0.0)
    ap.add_argument("--keep_ckpts", type=int, default=1,
                    help="save numbered checkpoints in the 2nd half for best-on-bench selection")
    ap.add_argument("--out", default="artifacts/sanity")
    a = ap.parse_args()

    device = pick_device()
    print(f"device={device} config={a.config}")
    tok = load_tokenizer(a.tok)
    cfg = CONFIGS[a.config]
    cfg.vocab_size = tok.get_vocab_size()

    data_dir = Path(a.data)
    train_ds = JQDataset(data_dir / "train.jsonl", tok, cfg.max_seq_len, context_loss_weight=a.ctx_loss)
    val_ds = JQDataset(data_dir / "val.jsonl", tok, cfg.max_seq_len)
    dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                    collate_fn=train_ds.collate, drop_last=True)
    print(f"train={len(train_ds)} val={len(val_ds)} params≈{cfg.n_params()/1e6:.1f}M")

    model = NL2JQModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    # seen-schema extraction subset (the M0 gate) + a train sample for overfit check
    train_rows = train_ds.rows
    extraction = [r for r in train_rows if r["tags"][0] in ("path", "pluck", "index")]
    print(f"seen-extraction eval pool: {len(extraction)}")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    log = []
    step = 0
    t0 = time.time()
    it = iter(dl)
    model.train()
    while step < a.steps:
        try:
            X, Y, W = next(it)
        except StopIteration:
            it = iter(dl)
            X, Y, W = next(it)
        X, Y, W = X.to(device), Y.to(device), W.to(device)
        lr = cosine_lr(step, a.warmup, a.steps, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        # bf16 autocast on CUDA: ~half the activation memory, ~2x faster on the 5090.
        # bf16 has fp32's exponent range, so no GradScaler is needed.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            _, loss = model(X, targets=Y, loss_mask=W)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 50 == 0:
            print(f"step {step:5d} loss {loss.item():.4f} lr {lr:.2e} "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")
        if step % a.eval_every == 0 or step == a.steps:
            val = eval_exec(model, tok, val_ds.rows, device, limit=a.eval_n)
            ext = eval_exec(model, tok, extraction, device, limit=a.eval_n)
            tr = eval_exec(model, tok, train_rows, device, limit=120)
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "train_exec": round(tr["exec_acc"], 3),
                   "val_exec": round(val["exec_acc"], 3),
                   "seen_extraction_exec": round(ext["exec_acc"], 3),
                   "val_valid_rate": round(val["valid_rate"], 3)}
            log.append(rec)
            print("  EVAL", json.dumps(rec))
            if val["fails"]:
                f = val["fails"][0]
                print(f"    e.g. req={f['request']!r}\n         gold={f['gold']}\n         pred={f['pred']}")
            # periodic checkpoint so a long unattended run survives a crash/preemption
            payload = {"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}
            torch.save(payload, Path(a.out) / "model.pt")
            # Keep NUMBERED checkpoints too: val_exec (own held-out families) is a mirage
            # — the real metric is the independent bench, scored post-hoc. Keeping every
            # eval's weights lets us pick the best-on-bench step instead of the last one
            # (v3's final step overfit its distribution and scored *worse* on the bench).
            if a.keep_ckpts and step >= a.steps // 3:
                torch.save(payload, Path(a.out) / f"model_{step}.pt")
            (Path(a.out) / "trainlog.json").write_text(json.dumps(log, indent=2))

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, Path(a.out) / "model.pt")
    (Path(a.out) / "trainlog.json").write_text(json.dumps(log, indent=2))
    final = log[-1]
    gate = final["train_exec"] >= 0.95 and final["seen_extraction_exec"] >= 0.90
    print(f"\nM0 GATE: {'PASS' if gate else 'FAIL'}  "
          f"train_exec={final['train_exec']} seen_extraction={final['seen_extraction_exec']}")
    print(f"generalization: val_exec={final['val_exec']}")


if __name__ == "__main__":
    main()
