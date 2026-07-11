"""Convert a trained nl2jq checkpoint (train/model.py NL2JQModel) into a standard
Hugging Face LlamaForCausalLM, so it loads with `transformers` and converts to GGUF
via llama.cpp with no custom code.

The architecture is already Llama-shaped and — crucially — uses the HF/GPT-NeoX RoPE
convention (rotate_half over first/second halves, cos = cat(freqs, freqs)), so the port
is a pure key rename with no weight permutation. We verify that by asserting logit parity
between the original module and the converted HF model on random inputs before saving.

    python -m release.to_hf --ckpt artifacts/nl2jq-40m/model.pt \
        --tok artifacts/tok --out artifacts/nl2jq-40m-hf
"""
import argparse
import json
from pathlib import Path

import torch

from train.model import CONFIGS, ModelConfig, NL2JQModel

# our state-dict key -> HF key template ({i} = layer index)
LAYER_MAP = {
    "n1.weight": "model.layers.{i}.input_layernorm.weight",
    "attn.q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "attn.k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "attn.v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "attn.o.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "n2.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "mlp.w1.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "mlp.w3.weight": "model.layers.{i}.mlp.up_proj.weight",
    "mlp.w2.weight": "model.layers.{i}.mlp.down_proj.weight",
}


def remap(sd, n_layer):
    out = {"model.embed_tokens.weight": sd["tok_emb.weight"],
           "model.norm.weight": sd["norm.weight"],
           "lm_head.weight": sd["lm_head.weight"]}
    for i in range(n_layer):
        for suffix, tmpl in LAYER_MAP.items():
            out[tmpl.format(i=i)] = sd[f"blocks.{i}.{suffix}"]
    return out


def to_llama_config(cfg: ModelConfig, eos_id, pad_id):
    from transformers import LlamaConfig
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        num_key_value_heads=cfg.n_kv_head,
        head_dim=cfg.d_model // cfg.n_head,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=1e-5,          # NL2JQModel RMSNorm default; HF default is 1e-6
        rope_theta=cfg.rope_theta,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", action="store_true",
                    help="skip loading ckpt; use a random 40m model (parity self-test only)")
    a = ap.parse_args()
    if not a.random and not a.ckpt:
        ap.error("--ckpt is required unless --random")

    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    if a.random:
        cfg = CONFIGS["40m"]
        cfg.vocab_size = 12288
        src = NL2JQModel(cfg).eval()
        sd = src.state_dict()
    else:
        blob = torch.load(a.ckpt, map_location="cpu")
        cfg = ModelConfig(**blob["cfg"])
        src = NL2JQModel(cfg).eval()
        src.load_state_dict(blob["model"])
        sd = blob["model"]

    tok_dir = Path(a.tok)
    meta = json.loads((tok_dir / "tok_meta.json").read_text()) if (tok_dir / "tok_meta.json").exists() else {}
    eos_id = meta.get("special_ids", {}).get("<|end|>", 4)
    pad_id = meta.get("special_ids", {}).get("<|pad|>", 0)

    hf_cfg = to_llama_config(cfg, eos_id, pad_id)
    hf = LlamaForCausalLM(hf_cfg).eval()
    missing, unexpected = hf.load_state_dict(remap(sd, cfg.n_layer), strict=False)
    # only the tied lm_head may show as 'missing' depending on HF version; embed drives it
    assert not unexpected, f"unexpected keys: {unexpected}"

    # --- logit parity: the real correctness check (covers RoPE convention) ---
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a_logits, _ = src(ids)
        b_logits = hf(ids).logits
    diff = (a_logits - b_logits).abs().max().item()
    print(f"max|logit diff| = {diff:.3e}")
    assert diff < 1e-3, f"parity FAILED: {diff}"
    print("parity OK — HF model matches the original module")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    hf.to(torch.bfloat16).save_pretrained(a.out)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_dir / "tokenizer.json"),
        eos_token="<|end|>", pad_token="<|pad|>",
        additional_special_tokens=["<|request|>", "<|input|>", "<|program|>"])
    hf_tok.save_pretrained(a.out)
    print(f"saved HF model + tokenizer -> {a.out}")


if __name__ == "__main__":
    main()
