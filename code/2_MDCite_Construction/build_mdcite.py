"""Construct the MDCite single-intent citation-context dataset.

Reads the ``citing_contexts.json`` artifacts produced by Pipeline 1 (either a
directory or the ``Citation context & intent data`` ZIP archive) and builds the
released MDCite tables:

    dataset_context_intent.parquet / .csv          multi-intent (provenance)
    dataset_context_intent_single.parquet / .csv   single-intent benchmark

Each output row is one citation context with the schema:
``text, label, field, group_id, paperId, doi, venue, year, source_file``.

The single-intent variant keeps only rows whose label is a single canonical
intent (labels containing a space, i.e. composite/multi-intent labels, are
dropped), matching the released MDCite benchmark.

Usage
-----
    # From the ZIP archive
    python build_mdcite.py --context-zip "Citation context & intent data.zip" --out-dir mdcite_out

    # From an already-extracted directory
    python build_mdcite.py --context-dir citation_context_intent_data --out-dir mdcite_out
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

CONTEXT_FILE = "citing_contexts.json"


def infer_field(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("output_"):
            return part
    return "Unknown"


def normalize_text(contexts) -> Optional[str]:
    if contexts is None:
        return None
    if isinstance(contexts, str):
        t = contexts.strip()
        return t or None
    if isinstance(contexts, list):
        sents = []
        for c in contexts:
            if isinstance(c, str):
                sents.append(c.strip())
            elif isinstance(c, dict):
                for k in ("text", "sentence", "context"):
                    if isinstance(c.get(k), str):
                        sents.append(c[k].strip())
                        break
        text = " ".join(s for s in sents if s).strip()
        return text or None
    return None


def get_label(intents) -> Optional[str]:
    if intents is None:
        return None
    if isinstance(intents, str):
        t = intents.strip()
        return t or None
    if isinstance(intents, list) and intents and isinstance(intents[0], str):
        t = intents[0].strip()  # policy: first (primary) intent
        return t or None
    return None


def record_to_row(r: dict, field: str, source: str) -> Optional[dict]:
    text = normalize_text(r.get("contexts"))
    label = get_label(r.get("intents"))
    if text is None or label is None:
        return None
    group_id = (
        r.get("paperId")
        or r.get("doi")
        or r.get("citing_doi")
        or r.get("citing_paper_doi")
        or "NA"
    )
    return {
        "text": text,
        "label": label,
        "field": field,
        "group_id": group_id,
        "paperId": r.get("paperId", ""),
        "doi": r.get("doi", ""),
        "venue": r.get("venue", ""),
        "year": r.get("year", ""),
        "source_file": source,
    }


def iter_records(raw: bytes) -> Iterable[dict]:
    try:
        obj = json.loads(raw)
    except Exception:
        return
    if isinstance(obj, list):
        for r in obj:
            if isinstance(r, dict):
                yield r
    elif isinstance(obj, dict):
        yield obj


def rows_from_dir(context_dir: str):
    files = sorted(Path(context_dir).rglob(CONTEXT_FILE))
    print(f"Found {len(files)} {CONTEXT_FILE} files.")
    for fp in files:
        field = infer_field(str(fp))
        raw = fp.read_bytes()
        for r in iter_records(raw):
            row = record_to_row(r, field, str(fp))
            if row:
                yield row


def rows_from_zip(zip_path: str):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("/" + CONTEXT_FILE)]
        print(f"Found {len(names)} {CONTEXT_FILE} entries in ZIP.")
        for name in names:
            field = infer_field(name)
            with z.open(name) as f:
                raw = f.read()
            for r in iter_records(raw):
                row = record_to_row(r, field, name)
                if row:
                    yield row


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the MDCite dataset.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--context-zip", help="Citation context & intent data ZIP.")
    src.add_argument("--context-dir", help="Extracted citing_contexts.json directory.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument(
        "--no-csv", action="store_true", help="Only write Parquet (skip large CSVs)."
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gen = rows_from_zip(args.context_zip) if args.context_zip else rows_from_dir(
        args.context_dir
    )

    df = pd.DataFrame(list(gen))
    print("Multi-intent rows:", df.shape)
    if df.empty:
        raise SystemExit("No citation-context records found.")

    multi_parquet = os.path.join(args.out_dir, "dataset_context_intent.parquet")
    df.to_parquet(multi_parquet, index=False)
    if not args.no_csv:
        df.to_csv(os.path.join(args.out_dir, "dataset_context_intent.csv"), index=False)

    df_single = df[~df["label"].str.contains(" ", na=False)].copy()
    print("Single-intent rows:", df_single.shape)
    single_parquet = os.path.join(args.out_dir, "dataset_context_intent_single.parquet")
    df_single.to_parquet(single_parquet, index=False)
    if not args.no_csv:
        df_single.to_csv(
            os.path.join(args.out_dir, "dataset_context_intent_single.csv"), index=False
        )

    print("\nLabel distribution (single):")
    print(df_single["label"].value_counts())
    print("\nSaved to:", args.out_dir)


if __name__ == "__main__":
    main()
