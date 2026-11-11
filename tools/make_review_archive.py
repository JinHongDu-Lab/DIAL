"""Build the anonymized code archive for double-blind review.

    python tools/make_review_archive.py [--out DIAL_code.zip]

The archive holds the files tracked at HEAD (code, configs, notebooks, prepared benchmarks, and
judge responses; no result files), with everything that could identify the authors removed:

  * `.github/`, `data/huggingface/` (dataset card, upload script), and this `tools/` directory
  * badges and links to the public repository, preprint, and dataset mirror in the READMEs
  * notebook outputs, execution counts, and kernel metadata
  * provider batch and response identifiers in `responses.parquet` (blanked; no analysis reads them)
  * file timestamps and owners (fixed in the zip)

The build fails if any identifying pattern remains in the archived files.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "DIAL"
EXCLUDE = (".github/", "data/huggingface/", "tools/")
BLANK_COLUMNS = ("provider_response_id", "provider_batch_id")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".sh", ".ipynb", ".example", ".cfg", ""}

# identities, accounts, local paths, links to public copies, and review-process vocabulary
FORBIDDEN = re.compile(
    r"jinhong|dujinhong|JinHongDu|\bCai\b|Zesheng|Yingqi|Sichang|\bCSML\b|andrew\.cmu|cmu\.edu|"
    r"/Users/|/home/[a-z]|arxiv\.org/abs/26|2609\.31215|huggingface\.co/datasets/(?!lmsys)|"
    r"github\.com/(?!WeOpenML|TanXZfra|github/|David-OConnor|pdm-project)|"
    r"rebuttal|reviewer|referee|\bR[123][WQ][0-9]\b|TODO|FIXME|XXX",
    re.IGNORECASE)


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [f for f in out.decode().split("\0") if f and not f.startswith(EXCLUDE)]


def clean_readme(text):
    """Drop the badge block and the paragraph on the public dataset mirror."""
    lines = [l for l in text.splitlines() if not l.startswith("[![")]
    text = "\n".join(lines) + "\n"
    text = re.sub(r"\nThe data are also on the Hugging Face Hub as\n.*?\n\n", "\n\n", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_data_readme(text):
    text = re.sub(r"^huggingface/ .*\n", "", text, flags=re.M)
    text = re.sub(r"The same files are mirrored on the Hugging Face Hub as\n.*?(?=\n## Benchmarks)", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_notebook(text):
    nb = json.loads(text)
    for cell in nb["cells"]:
        cell.pop("id", None)
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def clean_responses(path):
    """The responses table with the provider identifiers blanked, as parquet bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    for name in BLANK_COLUMNS:
        if name in table.column_names:
            i = table.column_names.index(name)
            table = table.set_column(i, name, pa.nulls(table.num_rows, type=table.schema.field(name).type))
    table = table.replace_schema_metadata(None)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def payload(rel):
    path = ROOT / rel
    if rel == "README.md":
        return clean_readme(path.read_text()).encode()
    if rel == "data/README.md":
        return clean_data_readme(path.read_text()).encode()
    if rel.endswith(".ipynb"):
        return clean_notebook(path.read_text()).encode()
    if rel.endswith("responses.parquet"):
        return clean_responses(path)
    return path.read_bytes()


def audit(rel, data):
    """Identifying strings left in one archived file."""
    if rel.endswith(".parquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(data))
        skip = {"prompt", "response_a", "response_b", "raw_response"}        # benchmark and model text
        hits = []
        if table.schema.metadata:
            hits += FORBIDDEN.findall(str(table.schema.metadata))
        for name in table.column_names:
            if name in skip or not str(table.schema.field(name).type).startswith(("string", "large_string")):
                continue
            for value in table.column(name).unique().to_pylist():
                if value and FORBIDDEN.search(value):
                    hits.append(f"{name}: {value[:80]}")
        return hits
    if Path(rel).suffix not in TEXT_SUFFIXES and Path(rel).name not in ("LICENSE", ".gitignore"):
        return []
    return [f"{n}: {line.strip()[:120]}" for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1) if FORBIDDEN.search(line)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "DIAL_code.zip"))
    a = p.parse_args()
    files, problems = {}, {}
    for rel in sorted(tracked_files()):
        files[rel] = payload(rel)
        hits = audit(rel, files[rel])
        if hits:
            problems[rel] = hits
    if problems:
        for rel, hits in problems.items():
            print(rel)
            for h in hits[:10]:
                print("   ", h)
        raise SystemExit("identifying strings remain; nothing written")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel, data in files.items():
            info = zipfile.ZipInfo(f"{TOP}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if rel.endswith(".parquet") else zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel.endswith(".sh") else 0o644) << 16
            z.writestr(info, data)
    size = Path(a.out).stat().st_size / 1e6
    print(f"{len(files)} files, {size:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()
