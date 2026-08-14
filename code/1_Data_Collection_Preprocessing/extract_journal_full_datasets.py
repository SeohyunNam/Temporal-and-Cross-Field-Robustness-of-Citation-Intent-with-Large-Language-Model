"""Split the merged field/journal table into per-journal ``*_full.csv`` files.

Second preprocessing step of the IDCite construction pipeline. It takes the
merged table produced by ``build_field_journal_mapping.py`` and writes one
CSV per (group, journal) pair, using the file-name convention consumed by
``select_top5pct_per_journal.py``::

    <out-dir>/<group_key>__<journal_key>_full.csv

Optionally, a per-group full CSV (all journals of a group concatenated) is
also written when ``--also-per-group`` is set.

Usage
-----
    python extract_journal_full_datasets.py \
        --merged-csv WOS_merged_preprocessed.csv \
        --out-dir Field_full_datasets_by_journal
"""

import argparse
import os
import re

import pandas as pd


def safe_key(name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(name))
    return key.strip("_") or "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split merged field/journal table into per-journal full CSVs."
    )
    ap.add_argument("--merged-csv", required=True, help="Merged CSV from step 1.")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for <group>__<journal>_full.csv files.",
    )
    ap.add_argument(
        "--also-per-group",
        action="store_true",
        help="Additionally write one concatenated CSV per group.",
    )
    ap.add_argument(
        "--per-group-dir",
        default=None,
        help="Directory for per-group CSVs (defaults to <out-dir>_by_group).",
    )
    args = ap.parse_args()

    merged = pd.read_csv(args.merged_csv, low_memory=False)
    print("merged shape:", merged.shape)

    os.makedirs(args.out_dir, exist_ok=True)
    groups = sorted(merged["group"].dropna().unique())
    print("groups:", len(groups))

    n_files = 0
    for g in groups:
        grp = merged[merged["group"] == g].copy()
        group_key = safe_key(g)

        journals = sorted(grp["journal_clarivate"].dropna().unique())
        for j in journals:
            sub = grp[grp["journal_clarivate"] == j].copy()
            if len(sub) == 0:
                continue
            out_path = os.path.join(
                args.out_dir, f"{group_key}__{safe_key(j)}_full.csv"
            )
            sub.to_csv(out_path, index=False)
            n_files += 1

    print(f"Saved {n_files} per-journal full CSVs to {args.out_dir}")

    if args.also_per_group:
        per_group_dir = args.per_group_dir or (args.out_dir.rstrip("/\\") + "_by_group")
        os.makedirs(per_group_dir, exist_ok=True)
        for g in groups:
            grp = merged[merged["group"] == g].copy()
            if len(grp) == 0:
                continue
            out_path = os.path.join(
                per_group_dir, f"{safe_key(g)}_all_top5journals_full.csv"
            )
            grp.to_csv(out_path, index=False)
        print(f"Saved per-group full CSVs to {per_group_dir}")


if __name__ == "__main__":
    main()
