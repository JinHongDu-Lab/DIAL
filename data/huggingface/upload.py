"""Publish the DIAL data to the Hugging Face Hub.

The Hub repository mirrors this ``data/`` directory: the prepared benchmarks, the collected judge
responses, each study's design and prompts, and the judge catalog.  The collection code stays on
GitHub.  ``huggingface/README.md`` becomes the dataset card.

    hf auth login                                     # once, with a write token
    python huggingface/upload.py --stage /tmp/dial-hf # write the layout locally, upload nothing
    python huggingface/upload.py --repo-id CSML/DIAL --private
    python huggingface/upload.py --repo-id CSML/DIAL   # public release
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
CARD = DATA_DIR / "huggingface" / "README.md"
INCLUDE = (
    "*/data/*.parquet",
    "*/responses.parquet",
    "*/study.toml",
    "*/prompts/*.txt",
    "configs/*.json",
)


def files() -> list[Path]:
    paths = [
        p
        for p in sorted(DATA_DIR.rglob("*"))
        if p.is_file() and any(fnmatch.fnmatch(str(p.relative_to(DATA_DIR)), g) for g in INCLUDE)
    ]
    if not paths:
        raise SystemExit(f"no data files under {DATA_DIR}")
    return paths


def stage(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CARD, target / "README.md")
    for p in files():
        dest = target / p.relative_to(DATA_DIR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    print(f"staged {len(files()) + 1} files under {target}")


def upload(repo_id: str, private: bool, message: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(CARD))]
    ops += [
        CommitOperationAdd(path_in_repo=str(p.relative_to(DATA_DIR)), path_or_fileobj=str(p))
        for p in files()
    ]
    for op in ops:
        print(f"  {op.path_in_repo}")
    info = api.create_commit(repo_id, repo_type="dataset", operations=ops, commit_message=message)
    print(f"uploaded {len(ops)} files: {info.commit_url}")
    print(f'\nfrom datasets import load_dataset\nload_dataset("{repo_id}", "mt_bench_responses")')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="CSML/DIAL", help="Hub dataset id (namespace/name)")
    parser.add_argument("--private", action="store_true", help="create the repository as private")
    parser.add_argument("--message", default="Release the DIAL judge panel", help="commit message")
    parser.add_argument("--stage", type=Path, help="copy the Hub layout to this directory instead of uploading")
    args = parser.parse_args()
    if args.stage:
        stage(args.stage)
    else:
        upload(args.repo_id, args.private, args.message)


if __name__ == "__main__":
    main()
