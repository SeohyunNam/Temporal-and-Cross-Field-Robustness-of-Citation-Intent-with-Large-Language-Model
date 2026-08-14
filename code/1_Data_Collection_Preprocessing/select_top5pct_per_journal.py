"""Select the top-5% most-cited seed papers, independently within each journal.

Third preprocessing step of the IDCite construction pipeline. For every
``<group>__<journal>_full.csv`` file produced by
``extract_journal_full_datasets.py`` it selects the top 5% most-cited papers
(computed on positively-cited papers only, per journal), then aggregates the
selected papers per group.

Selecting the threshold independently within each journal implements the
journal-stratified strategy that avoids over-representing citation-intensive
fields.

For each group it writes:

    <group_key>_top5pct_per_journal_for_paper_title.csv   title + doi (+ meta)
    <group_key>_top5pct_per_journal_full.csv              full metadata
    <group_key>_top5pct_per_journal_full.json            full metadata (JSON)

and a global ``top5pct_per_journal_summary.csv``. The ``*_for_paper_title.csv``
files are the seed/anchor input to ``batch_paper_title_multi.py``.

Usage
-----
    python select_top5pct_per_journal.py \
        --journal-full-dir Field_full_datasets_by_journal \
        --out-dir Field_top5pct_per_journal \
        --pct 0.95
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

CITED_COL = "citedby_count"


def safe_key(name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(name))
    return key.strip("_") or "UNKNOWN"


def parse_group_journal(path: str):
    base = os.path.basename(path)
    name = base.replace("_full.csv", "")
    if "__" not in name:
        return None, None
    group_key, journal_key = name.split("__", 1)
    return group_key, journal_key


def select_top(sub_pos: pd.DataFrame, quantile: float) -> pd.DataFrame:
    q = sub_pos[CITED_COL].quantile(quantile)
    top = sub_pos[sub_pos[CITED_COL] >= q].copy()
    if len(top) == 0:  # numeric fallback
        k = max(1, int(len(sub_pos) * (1 - quantile)))
        top = sub_pos.sort_values(CITED_COL, ascending=False).head(k).copy()
    return top.sort_values(CITED_COL, ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select top-5% cited seed papers per journal, aggregated per group."
    )
    ap.add_argument(
        "--journal-full-dir",
        required=True,
        help="Directory of <group>__<journal>_full.csv files (from step 2).",
    )
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument(
        "--pct",
        type=float,
        default=0.95,
        help="Quantile threshold (default 0.95 = top 5%).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    journal_files = sorted(glob.glob(os.path.join(args.journal_full_dir, "*_full.csv")))
    if not journal_files:
        raise SystemExit(f"No *_full.csv files found in {args.journal_full_dir}")
    print(f"Found {len(journal_files)} per-journal full files.")

    group_top_parts = {}
    group_name_map = {}
    summary_rows = []

    for csv_path in journal_files:
        group_key, journal_key = parse_group_journal(csv_path)
        if not group_key or not journal_key:
            print("[warn] cannot parse file name -> skip:", csv_path)
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        if CITED_COL not in df.columns:
            print(f"[warn] no '{CITED_COL}' in {csv_path} -> skip")
            continue
        df[CITED_COL] = pd.to_numeric(df[CITED_COL], errors="coerce")

        group_name = str(df["group"].iloc[0]) if "group" in df.columns else group_key
        journal_name = (
            str(df["journal_clarivate"].iloc[0])
            if "journal_clarivate" in df.columns
            else journal_key
        )
        group_name_map[group_key] = group_name

        sub = df.dropna(subset=[CITED_COL])
        sub_pos = sub[sub[CITED_COL] > 0].copy()
        if len(sub_pos) == 0:
            print(f"[{group_name}] {journal_name}: no positively-cited papers -> skip")
            continue

        top_j = select_top(sub_pos, args.pct)
        summary_rows.append(
            {
                "group": group_name,
                "group_key": group_key,
                "journal_clarivate": journal_name,
                "journal_key": journal_key,
                "n_total_journal": len(sub),
                "n_pos_cited": len(sub_pos),
                "n_top5pct_journal": len(top_j),
                "cited_threshold": float(sub_pos[CITED_COL].quantile(args.pct)),
            }
        )
        group_top_parts.setdefault(group_key, []).append(top_j)

    for group_key, parts in group_top_parts.items():
        group_name = group_name_map.get(group_key, group_key)
        top_all = pd.concat(parts, ignore_index=True).sort_values(
            CITED_COL, ascending=False
        )
        print(f"[{group_name}] aggregated top-5% papers = {len(top_all)}")

        cols_for_batch = [
            c
            for c in ["title", "doi", CITED_COL, "group", "journal_clarivate"]
            if c in top_all.columns
        ]
        if "title" not in cols_for_batch or "doi" not in cols_for_batch:
            raise SystemExit(
                f"[{group_name}] missing title/doi columns for paper-title batch input."
            )

        base = os.path.join(args.out_dir, f"{group_key}_top5pct_per_journal")
        top_all[cols_for_batch].to_csv(base + "_for_paper_title.csv", index=False)
        top_all.to_csv(base + "_full.csv", index=False)
        with open(base + "_full.json", "w", encoding="utf-8") as f:
            json.dump(top_all.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.out_dir, "top5pct_per_journal_summary.csv"), index=False
        )
    print("Done. Output in:", args.out_dir)


if __name__ == "__main__":
    main()
