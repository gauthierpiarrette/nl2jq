"""Load JSONL examples into padded (input, target, loss_mask) tensors.

Loss is computed only on program tokens (from the token after <|program|> through
<|end|> inclusive); request/input tokens are masked out for the sanity model.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JQDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len, program_token="<|program|>",
                 eos="<|end|>", context_loss_weight=0.0):
        self.rows = [json.loads(l) for l in Path(jsonl_path).open()]
        self.tok = tokenizer
        self.max_len = max_len
        self.prog_id = tokenizer.token_to_id(program_token)
        self.eos_id = tokenizer.token_to_id(eos)
        self.ctx_w = context_loss_weight
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        # pre-tokenize once (batch encode) so epochs don't re-encode in Python
        encs = tokenizer.encode_batch([r["text"] for r in self.rows])
        self.cache = [e.ids[: max_len] for e in encs]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = self.cache[i]
        try:
            p = ids.index(self.prog_id)
        except ValueError:
            p = len(ids) - 1
        x = ids[:-1]
        y = ids[1:]
        # weight targets: program region weight 1.0, context weight ctx_w
        w = [self.ctx_w] * len(y)
        for j in range(len(y)):
            if j >= p:  # target y[j] = ids[j+1] is a program-region token
                w[j] = 1.0
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(w, dtype=torch.float))

    def collate(self, batch):
        maxlen = max(len(x) for x, _, _ in batch)
        X, Y, W = [], [], []
        for x, y, w in batch:
            pad = maxlen - len(x)
            X.append(torch.cat([x, torch.full((pad,), self.pad_id)]))
            Y.append(torch.cat([y, torch.full((pad,), -100)]))
            W.append(torch.cat([w, torch.zeros(pad)]))
        return torch.stack(X), torch.stack(Y), torch.stack(W)
