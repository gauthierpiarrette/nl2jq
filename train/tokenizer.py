"""Train / load the nl2jq BPE tokenizer.

Design (SPEC §5.1): byte-level fallback (any field name is representable), single-digit
number tokens (reliable numeric copying), and the four structural special tokens.

    python -m train.tokenizer --data data/v0 --vocab 8192 --out artifacts/tok
"""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

from pipeline.common import SPECIAL_TOKENS


def train_tokenizer(data_dir: Path, vocab_size: int, out_dir: Path):
    tok = Tokenizer(BPE(unk_token=None, fuse_unk=False, byte_fallback=True))
    # Split digit runs into single digits, then byte-level everything (GPT-2 alphabet).
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    def corpus():
        for split in ("train.jsonl", "val.jsonl"):
            fp = data_dir / split
            if fp.exists():
                for line in fp.open():
                    yield json.loads(line)["text"]

    tok.train_from_iterator(corpus(), trainer=trainer)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    meta = {"vocab_size": tok.get_vocab_size(),
            "special_tokens": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}}
    (out_dir / "tok_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"trained tokenizer: {tok.get_vocab_size()} tokens -> {out_dir}")
    print("special ids:", meta["special_tokens"])
    return tok


def load_tokenizer(out_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v0")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--out", default="artifacts/tok")
    a = ap.parse_args()
    train_tokenizer(Path(a.data), a.vocab, Path(a.out))


if __name__ == "__main__":
    main()
