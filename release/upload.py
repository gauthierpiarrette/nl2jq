"""Push artifacts to the Hugging Face Hub. Requires HF_TOKEN and `pip install huggingface_hub`.

    python -m release.upload dataset --repo gauthierpiarrette/nl2jq --data data/v2
    python -m release.upload model   --repo gauthierpiarrette/nl2jq-40m --model artifacts/nl2jq-40m
"""
import argparse
import os
from pathlib import Path

REL = Path(__file__).resolve().parent


def upload_dataset(repo, data_dir, private=True):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)
    api.upload_file(path_or_fileobj=str(REL / "DATASET_CARD.md"),
                    path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    # prefer gzipped splits (jsonl.gz is HF-viewer compatible and ~7x smaller)
    for split in ("train.jsonl.gz", "val.jsonl.gz", "train.jsonl", "val.jsonl"):
        fp = Path(data_dir) / split
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"data/{split}",
                            repo_id=repo, repo_type="dataset")
            print(f"  + {split}")
    print(f"uploaded dataset -> https://huggingface.co/datasets/{repo}")


def upload_bench(repo, private=True):
    """The bench dataset repo: BENCH_CARD as README + frozen v1.0.0 + FREEZE record +
    the demoted dev split + the standalone harness. (The sealed canary set is never
    uploaded anywhere.)"""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)
    bench = REL.parent / "bench"
    api.upload_file(path_or_fileobj=str(REL / "BENCH_CARD.md"),
                    path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    ups = [(bench / "frozen" / "nl2jq-bench-1.0.0.jsonl", "nl2jq-bench-1.0.0.jsonl"),
           (bench / "frozen" / "FREEZE.txt", "FREEZE.txt"),
           (bench / "nl2jq-bench.jsonl", "devset-v0.jsonl"),
           (bench / "harness.py", "harness.py")]
    for src, dst in ups:
        if src.exists():
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                            repo_id=repo, repo_type="dataset")
            print(f"  + {dst}")
    # the pre-v1.0 filename is superseded by devset-v0.jsonl — remove if present
    try:
        api.delete_file("nl2jq-bench.jsonl", repo_id=repo, repo_type="dataset")
        print("  - nl2jq-bench.jsonl (renamed to devset-v0.jsonl)")
    except Exception:
        pass
    print(f"uploaded bench -> https://huggingface.co/datasets/{repo}")


def upload_model(repo, model_dir, private=True, card="MODEL_CARD.md"):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    card_path = card if os.path.isabs(card) else str(REL / card)
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=repo)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo,
                      ignore_patterns=["*.log", "trainlog.json", "*.pt"])
    print(f"uploaded model -> https://huggingface.co/{repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dataset", "bench", "model"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--model", default="artifacts/nl2jq-40m")
    ap.add_argument("--card", default="MODEL_CARD.md",
                    help="model card to publish as README.md (relative to release/ or absolute)")
    ap.add_argument("--public", action="store_true", help="create a public repo (default private)")
    a = ap.parse_args()
    if a.kind == "dataset":
        upload_dataset(a.repo, a.data, private=not a.public)
    elif a.kind == "bench":
        upload_bench(a.repo, private=not a.public)
    else:
        upload_model(a.repo, a.model, private=not a.public, card=a.card)


if __name__ == "__main__":
    main()
