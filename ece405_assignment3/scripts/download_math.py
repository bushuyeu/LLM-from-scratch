"""Download the MATH dataset from HuggingFace and split into train/validation JSONL.

Usage:
    uv run python scripts/download_math.py --output-dir /path/to/data/math
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path


def clear_stale_cache(dataset_name: str) -> None:
    """Delete any stale HF datasets Arrow cache for this dataset."""
    slug = dataset_name.replace("/", "___")
    candidates = [
        Path.home() / ".cache" / "huggingface" / "datasets" / slug,
        Path(os.environ.get("HF_DATASETS_CACHE", "~/.cache/huggingface/datasets")).expanduser() / slug,
    ]
    for path in candidates:
        if path.exists():
            shutil.rmtree(path)
            print(f"Cleared stale cache: {path}")


def download(output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    if train_path.exists() and val_path.exists():
        print(f"Dataset already present at {output_dir}, skipping download.")
        return

    # Bypass the `datasets` library entirely — it has a LocalFileSystem bug in 3.x
    # for this dataset's builder. Use huggingface_hub + pyarrow directly instead.
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow as pa
    import pyarrow.parquet as pq

    repo_id = "qwedsacf/competition_math"
    print(f"Listing parquet files in {repo_id} ...")
    parquet_files = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.endswith(".parquet")
    )
    print(f"Found: {parquet_files}")

    tables = []
    for filename in parquet_files:
        local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        tables.append(pq.read_table(local))

    table = pa.concat_tables(tables)
    col_dict = table.to_pydict()
    n = len(next(iter(col_dict.values())))
    rows = [{k: col_dict[k][i] for k in col_dict} for i in range(n)]
    print(f"Downloaded {n} examples total.")

    random.Random(seed).shuffle(rows)

    def write_jsonl(data, path):
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    write_jsonl(rows[:7500], train_path)
    write_jsonl(rows[7500:], val_path)
    print(f"Saved {len(rows[:7500])} train, {len(rows[7500:])} validation → {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    download(args.output_dir, args.seed)
