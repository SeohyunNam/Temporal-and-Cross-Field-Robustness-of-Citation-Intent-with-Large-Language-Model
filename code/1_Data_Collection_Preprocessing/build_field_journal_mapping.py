"""Match journal-stratified Scopus records to WoS field/category groups.

This is the first preprocessing step of the IDCite construction pipeline. It
takes the raw per-journal Scopus bibliographic records (collected with
``collect_by_journal.py``) together with a WoS / Journal Citation Reports group
definition and produces a single merged table in which every record is tagged
with its scientific field (group), representative WoS category, and canonical
journal name.

Inputs
------
--scopus-zip
    ZIP archive of raw Scopus records organized as
    ``<Group>/scopus_<JOURNAL>_2000_2024.csv``.
--group-file
    Plain-text WoS/JCR group definition. Records are read in fixed 7-line
    blocks::

        <group>
        <wos_category>
        <journal_1>
        <journal_2>
        <journal_3>
        <journal_4>
        <journal_5>

    Blank lines and the literal markers ``Group:``, ``Category:``,
    ``Journal:`` and any ``JCR Year`` header line are ignored.

Output
------
--out
    Merged CSV with the original Scopus columns plus ``group``, ``category``,
    ``journal_clarivate`` and ``journal_file_raw``. This file is the input to
    ``extract_journal_full_datasets.py``.

Usage
-----
    python build_field_journal_mapping.py \
        --scopus-zip "Scopus (Year 2000 - 2024).zip" \
        --group-file wos_groups.txt \
        --out WOS_merged_preprocessed.csv
"""

import argparse
import os
import re
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

SCOPUS_FILE_RE = re.compile(r"scopus_(.+?)_2000_2024\.csv")


def norm_key(s: str) -> str:
    """Uppercase alphanumeric-only key used for fuzzy name matching."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def parse_group_file(path: str) -> List[dict]:
    """Parse the WoS/JCR group definition into (group, category, journals)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    lines = [
        l.strip()
        for l in raw.splitlines()
        if l.strip() not in ("", "Group:", "Category:", "Journal:")
        and not l.startswith("JCR Year")
    ]

    blocks = []
    for i in range(0, len(lines), 7):
        chunk = lines[i : i + 7]
        if len(chunk) < 7:
            print("[warn] trailing lines ignored:", chunk)
            break
        blocks.append(
            {"group": chunk[0], "category": chunk[1], "journals": chunk[2:]}
        )
    return blocks


def list_zip_groups(zip_path: str) -> Dict[str, List[str]]:
    per_folder: Dict[str, List[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/", 1)
            if len(parts) != 2:
                continue
            folder, file = parts
            per_folder[folder].append(file)
    return per_folder


def match_group_folder(clar_group: str, zip_group_names) -> Optional[str]:
    """Match a group name to the best-fitting zip top-level folder."""
    k = norm_key(clar_group)
    for folder in zip_group_names:
        if norm_key(folder) == k:
            return folder

    best, best_score = None, -1
    for folder in zip_group_names:
        score = len(set(norm_key(folder)) & set(k))
        if score > best_score:
            best, best_score = folder, score
    return best


def get_group_sources(per_folder: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    group_sources: Dict[str, List[dict]] = defaultdict(list)
    for folder, files in per_folder.items():
        for file in files:
            base = os.path.basename(file)
            m = SCOPUS_FILE_RE.match(base)
            if not m:
                continue
            raw_name = m.group(1)
            group_sources[folder].append(
                {
                    "file": file,
                    "raw_name": raw_name,
                    "norm": norm_key(raw_name.replace("_", " ")),
                }
            )
    return group_sources


def match_journals_for_group(clar_block, folder, sources) -> List[dict]:
    canon_list = [{"name": j, "norm": norm_key(j)} for j in clar_block["journals"]]

    mapping = []
    for src in sources:
        best, best_score = None, -1
        for canon in canon_list:
            inter = len(set(src["norm"]) & set(canon["norm"]))
            bonus = 0
            if src["norm"] == canon["norm"]:
                bonus += 100
            elif src["norm"] in canon["norm"] or canon["norm"] in src["norm"]:
                bonus += 50
            score = inter + bonus
            if score > best_score:
                best, best_score = canon, score
        mapping.append(
            {
                "folder": folder,
                "file": src["file"],
                "raw_name": src["raw_name"],
                "clar_journal": best["name"] if best else src["raw_name"],
                "score": best_score,
            }
        )
    return mapping


def build_merged_dataset(zip_path: str, group_blocks: List[dict]) -> pd.DataFrame:
    per_folder = list_zip_groups(zip_path)
    group_sources = get_group_sources(per_folder)
    folder_map = {
        b["group"]: match_group_folder(b["group"], per_folder.keys())
        for b in group_blocks
    }

    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for block in group_blocks:
            group = block["group"]
            folder = folder_map[group]
            if folder is None:
                print(f"[warn] no zip folder matched for group: {group}")
                continue

            for m in match_journals_for_group(block, folder, group_sources[folder]):
                inner = f"{folder}/{os.path.basename(m['file'])}"
                with z.open(inner) as f:
                    df = pd.read_csv(f, low_memory=False)

                df["group"] = group
                df["category"] = block["category"]
                df["journal_clarivate"] = m["clar_journal"]
                df["journal_file_raw"] = m["raw_name"]
                rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Match Scopus per-journal records to WoS field/category groups."
    )
    ap.add_argument("--scopus-zip", required=True, help="Raw Scopus records ZIP.")
    ap.add_argument("--group-file", required=True, help="WoS/JCR group definition text.")
    ap.add_argument("--out", required=True, help="Output merged CSV path.")
    args = ap.parse_args()

    blocks = parse_group_file(args.group_file)
    print(f"Parsed {len(blocks)} field/category groups.")

    merged = build_merged_dataset(args.scopus_zip, blocks)
    if merged.empty:
        print("No records merged. Check inputs.")
        return

    merged.to_csv(args.out, index=False, encoding="utf-8")
    print(
        f"Saved: {args.out} | rows={len(merged)} | "
        f"groups={merged['group'].nunique()} | journals={merged['journal_clarivate'].nunique()}"
    )


if __name__ == "__main__":
    main()
