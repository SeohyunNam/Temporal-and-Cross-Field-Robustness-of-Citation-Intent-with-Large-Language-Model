"""Construct the EdgeCite retrieval-oriented citation-event benchmark.

Reorganizes the **Citation context & intent data** into a citation-event level
retrieval benchmark with normalized cited anchors, title-based candidate
representations, a leakage-controlled citing-disjoint split, and year metadata.

Pipeline (single script, mirroring the original multi-step notebook):

    1. Parse citing_contexts.json -> citation edges
       (citing_doi, cited_doi, field, context, primary_label).
       * cited_doi is decoded from the per-seed DOI folder name.
       * primary_label is the first canonical intent of each record.
    2. Keep rows with a non-empty context and a canonical primary label.
    3. Drop duplicates and boilerplate/noise contexts.
    4. Join the cited anchor title from the Top-5% seed CSVs; keep matched rows.
    5. Citing-disjoint train/validation/test split (80/10/10) for leakage control.
    6. Attach citing-paper year metadata.

Output
------
    retrieval_dataset_citing_disjoint_with_year.parquet

with columns: ``citing_doi, cited_doi_norm, cited_title, field, context,
primary_label, split, year``.

Usage
-----
    python build_edgecite.py \
        --context-zip "Citation context & intent data.zip" \
        --top5-zip "Top 5% cited papers per journal dataset.zip" \
        --out retrieval_dataset_citing_disjoint_with_year.parquet
"""

import argparse
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

CONTEXT_FILE = "citing_contexts.json"
CANONICAL = [
    "background",
    "motivation",
    "uses",
    "extends",
    "similarities",
    "differences",
    "future_work",
]

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)

_NOISE = re.compile(
    r"(all rights reserved|copyright|©|\bfigure\b|\btable\b|supplementary|"
    r"publisher's note|creative commons|license|doi:|covid-19 resource centre|"
    r"elsevier hereby grants permission|informed consent statement)",
    re.IGNORECASE,
)
_BRACKETS_ONLY = re.compile(r"^\s*\[\d+\]\.\s*$")
_DIGITS_PUNCT = re.compile(r"^[\d\W_]+$")


def norm_doi(x) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip().lower()
    if x in ("", "nan", "none", "na"):
        return None
    for p in _DOI_PREFIXES:
        x = x.replace(p, "")
    return x.replace("doi:", "").strip().rstrip(".") or None


def decode_doi_folder(folder: str) -> str:
    """'10.1002_jsfa.6703' -> '10.1002/jsfa.6703'."""
    if folder.startswith("10.") and "_" in folder:
        return folder.replace("_", "/", 1)
    return folder


def parse_path(zip_inner_path: str):
    parts = zip_inner_path.split("/")
    try:
        out_idx = next(i for i, p in enumerate(parts) if p.startswith("output_"))
    except StopIteration:
        return None, None
    field = parts[out_idx].replace("output_", "")
    cited_doi = decode_doi_folder(parts[out_idx + 1]) if out_idx + 1 < len(parts) else None
    return field, cited_doi


def parse_primary_label(intents) -> Optional[str]:
    if intents is None:
        return None
    if isinstance(intents, list):
        parts = [str(p).strip().lower() for p in intents]
    else:
        s = str(intents).strip().lower()
        if s in ("", "nan", "none", "na"):
            return None
        parts = s.split()
    for p in parts:
        if p in CANONICAL:
            return p
    return None


def normalize_context(contexts) -> Optional[str]:
    if isinstance(contexts, str):
        return contexts.strip() or None
    if isinstance(contexts, list):
        out = []
        for c in contexts:
            if isinstance(c, str):
                out.append(c.strip())
            elif isinstance(c, dict):
                for k in ("text", "sentence", "context"):
                    if isinstance(c.get(k), str):
                        out.append(c[k].strip())
                        break
        return (" ".join(x for x in out if x).strip()) or None
    return None


def iter_context_files(context_zip, context_dir):
    """Yield (inner_path, raw_bytes) for every citing_contexts.json."""
    if context_zip:
        with zipfile.ZipFile(context_zip) as z:
            names = [n for n in z.namelist() if n.endswith("/" + CONTEXT_FILE)]
            print(f"Found {len(names)} {CONTEXT_FILE} entries in ZIP.")
            for name in names:
                with z.open(name) as f:
                    yield name, f.read()
    else:
        files = sorted(Path(context_dir).rglob(CONTEXT_FILE))
        print(f"Found {len(files)} {CONTEXT_FILE} files.")
        for fp in files:
            yield str(fp), fp.read_bytes()


def build_edges(context_zip, context_dir):
    edges, year_rows = [], []
    for path, raw in iter_context_files(context_zip, context_dir):
        field, cited_doi = parse_path(path)
        cited_doi = norm_doi(cited_doi)
        try:
            records = json.loads(raw)
        except Exception:
            continue
        if not isinstance(records, list):
            records = [records]

        for r in records:
            if not isinstance(r, dict):
                continue
            citing_doi = norm_doi(r.get("doi"))
            if citing_doi is None:
                continue
            yr = r.get("year")
            if yr is not None:
                year_rows.append((citing_doi, yr))

            primary = parse_primary_label(r.get("intents"))
            context = normalize_context(r.get("contexts"))
            if primary is None or context is None:
                continue
            edges.append(
                {
                    "citing_doi": citing_doi,
                    "cited_doi_norm": cited_doi,
                    "field": field,
                    "context": context,
                    "primary_label": primary,
                }
            )

    df = pd.DataFrame(edges)
    year_map = (
        pd.DataFrame(year_rows, columns=["citing_doi", "year"])
        .drop_duplicates("citing_doi")
        if year_rows
        else pd.DataFrame(columns=["citing_doi", "year"])
    )
    return df, year_map


def clean_edges(df: pd.DataFrame, min_chars: int = 20) -> pd.DataFrame:
    df = df.dropna(subset=["context", "primary_label", "cited_doi_norm"]).copy()
    df = df[df["context"].str.strip().ne("")]
    df = df.drop_duplicates(
        subset=["citing_doi", "cited_doi_norm", "context", "primary_label"]
    )
    ctx = df["context"].astype(str)
    mask_noise = (
        (ctx.str.len() < min_chars)
        | ctx.str.match(_BRACKETS_ONLY)
        | ctx.str.match(_DIGITS_PUNCT)
        | ctx.str.contains(_NOISE)
    )
    return df[~mask_noise].copy()


def load_titles(top5_zip, top5_dir) -> pd.DataFrame:
    tables = []
    if top5_zip:
        with zipfile.ZipFile(top5_zip) as z:
            names = [n for n in z.namelist() if n.lower().endswith("_for_paper_title.csv")]
            for inner in names:
                with z.open(inner) as f:
                    tables.append(pd.read_csv(f, dtype=str))
    else:
        for fp in sorted(Path(top5_dir).rglob("*_for_paper_title.csv")):
            tables.append(pd.read_csv(fp, dtype=str))

    if not tables:
        raise SystemExit("No *_for_paper_title.csv anchor files found.")
    df_t = pd.concat(tables, ignore_index=True)
    df_t["cited_doi_norm"] = df_t["doi"].map(norm_doi)
    df_t = df_t.dropna(subset=["cited_doi_norm", "title"]).drop_duplicates("cited_doi_norm")
    return df_t[["cited_doi_norm", "title"]].rename(columns={"title": "cited_title"})


def citing_disjoint_split(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    citing = df["citing_doi"].dropna().unique()
    rng.shuffle(citing)
    n = len(citing)
    train = set(citing[: int(n * 0.80)])
    val = set(citing[int(n * 0.80) : int(n * 0.90)])
    df = df.copy()
    df["split"] = np.where(
        df["citing_doi"].isin(train),
        "train",
        np.where(df["citing_doi"].isin(val), "validation", "test"),
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the EdgeCite retrieval benchmark.")
    csrc = ap.add_mutually_exclusive_group(required=True)
    csrc.add_argument("--context-zip", help="Citation context & intent data ZIP.")
    csrc.add_argument("--context-dir", help="Extracted citing_contexts.json directory.")
    tsrc = ap.add_mutually_exclusive_group(required=True)
    tsrc.add_argument("--top5-zip", help="Top 5% cited papers per journal ZIP.")
    tsrc.add_argument("--top5-dir", help="Directory of *_for_paper_title.csv anchors.")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--seed", type=int, default=0, help="Split shuffle seed.")
    args = ap.parse_args()

    print("[1/6] Building edges from citation contexts...")
    df, year_map = build_edges(args.context_zip, args.context_dir)
    print("   raw edges:", df.shape)

    print("[2/6] Cleaning edges...")
    df = clean_edges(df)
    print("   clean edges:", df.shape)

    print("[3/6] Joining cited-anchor titles...")
    titles = load_titles(args.top5_zip, args.top5_dir)
    df = df.merge(titles, on="cited_doi_norm", how="left")
    df = df[df["cited_title"].notna()].copy()
    print("   title-matched edges:", df.shape)

    print("[4/6] Citing-disjoint split...")
    df = citing_disjoint_split(df, seed=args.seed)

    print("[5/6] Attaching year metadata...")
    df = df.merge(year_map, on="citing_doi", how="left")

    print("[6/6] Saving...")
    cols = [
        "citing_doi",
        "cited_doi_norm",
        "cited_title",
        "field",
        "context",
        "primary_label",
        "split",
        "year",
    ]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    df[cols].to_parquet(args.out, index=False)

    print("\nUsable citation edges:", len(df))
    print("Unique citing DOIs:", df["citing_doi"].nunique())
    print("Unique cited anchors:", df["cited_doi_norm"].nunique())
    print(df["split"].value_counts())
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
