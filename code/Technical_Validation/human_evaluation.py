"""Human evaluation of automatically assigned citation-intent labels.

This script reproduces the intent-annotation validation reported in the
IDCite paper. It compares model-predicted citation intents with expert
human labels stored in ``human_evaluation_Tuan_Anh_Phan.zip`` (Zenodo
Version 3).

The archive is organized as 21 disciplinary folders, each containing one
CSV file per canonical intent (21 x 7 = 147 files). Each CSV provides:

    * citation context
    * Prediction from classification model
    * Prediction from human

The script reports:

    1. Strict agreement (exact multi-label set match).
    2. Weak agreement (at least one overlapping intent).
    3. Micro- and macro-averaged precision, recall, and F1.
    4. Per-intent precision, recall, F1, and human support.
    5. Strict/weak agreement stratified by evaluation intent.

Usage
-----
    python human_evaluation.py --zip-path /path/to/human_evaluation_Tuan_Anh_Phan.zip
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

CANONICAL_INTENTS = {
    "background",
    "uses",
    "similarities",
    "differences",
    "motivation",
    "extends",
    "future_work",
}

LABEL_ORDER = [
    "background",
    "uses",
    "similarities",
    "differences",
    "motivation",
    "extends",
    "future_work",
]


def normalize_label_string(x):
    if pd.isna(x):
        return set()

    x = str(x).strip().lower()

    if x in {"", "nan", "none", "null"}:
        return set()

    x = x.replace("future work", "future_work")
    x = x.replace("future-work", "future_work")

    x = re.sub(r"[,;/|+]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return {
        token for token in x.split()
        if token in CANONICAL_INTENTS
    }


def load_evaluation_archive(zip_path: Path) -> pd.DataFrame:
    all_dfs = []

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]
        print("Number of CSV files:", len(csv_files))

        for file_name in csv_files:
            parts = file_name.split("/")
            discipline = parts[-2]
            base_name = os.path.basename(file_name)
            target_intent = base_name.replace("_human_evaluation.csv", "")

            with z.open(file_name) as f:
                df = pd.read_csv(f)

            df = df.rename(columns={
                "citation context": "citation_context",
                "Prediction from classsification model": "model_prediction",
                "Prediction from classification model": "model_prediction",
                "Prediction from human": "human_prediction",
            })

            df["discipline"] = discipline
            df["evaluation_intent"] = target_intent
            df["source_file"] = file_name
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print("Total evaluation records:", len(combined))
    print("Disciplines:", combined["discipline"].nunique())
    print("Evaluation intents:", combined["evaluation_intent"].nunique())
    return combined


def evaluate(combined: pd.DataFrame):
    combined = combined.copy()
    combined["model_set"] = combined["model_prediction"].apply(normalize_label_string)
    combined["human_set"] = combined["human_prediction"].apply(normalize_label_string)

    eval_df = combined[combined["human_set"].apply(len) > 0].copy()
    print("Total records:", len(combined))
    print("Valid human-evaluated records:", len(eval_df))
    print("Excluded records:", len(combined) - len(eval_df))

    eval_df["strict_match"] = eval_df["model_set"] == eval_df["human_set"]
    eval_df["weak_match"] = eval_df.apply(
        lambda row: len(row["model_set"] & row["human_set"]) > 0,
        axis=1,
    )

    strict_agreement = eval_df["strict_match"].mean()
    weak_agreement = eval_df["weak_match"].mean()
    print(f"Strict agreement: {strict_agreement:.4f} ({strict_agreement * 100:.2f}%)")
    print(f"Weak agreement:   {weak_agreement:.4f} ({weak_agreement * 100:.2f}%)")

    mlb = MultiLabelBinarizer(classes=LABEL_ORDER)
    mlb.fit([LABEL_ORDER])
    y_true = mlb.transform(eval_df["human_set"])
    y_pred = mlb.transform(eval_df["model_set"])

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")
    print()
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    intent_p, intent_r, intent_f1, intent_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    intent_results = pd.DataFrame({
        "Intent": mlb.classes_,
        "Human_Support": intent_support,
        "Precision": intent_p,
        "Recall": intent_r,
        "F1": intent_f1,
    })
    intent_results[["Precision", "Recall", "F1"]] = (
        intent_results[["Precision", "Recall", "F1"]].round(4)
    )

    by_intent = (
        eval_df.groupby("evaluation_intent")
        .agg(
            N=("strict_match", "size"),
            Strict_Agreement=("strict_match", "mean"),
            Weak_Agreement=("weak_match", "mean"),
        )
        .reset_index()
    )
    by_intent["Strict_Agreement"] = (by_intent["Strict_Agreement"] * 100).round(2)
    by_intent["Weak_Agreement"] = (by_intent["Weak_Agreement"] * 100).round(2)

    overall = pd.DataFrame(
        [
            {"Metric": "Valid human-evaluated records", "Value": len(eval_df)},
            {"Metric": "Strict agreement", "Value": round(strict_agreement, 4)},
            {"Metric": "Weak agreement", "Value": round(weak_agreement, 4)},
            {"Metric": "Micro Precision", "Value": round(micro_p, 4)},
            {"Metric": "Micro Recall", "Value": round(micro_r, 4)},
            {"Metric": "Micro F1", "Value": round(micro_f1, 4)},
            {"Metric": "Macro Precision", "Value": round(macro_p, 4)},
            {"Metric": "Macro Recall", "Value": round(macro_r, 4)},
            {"Metric": "Macro F1", "Value": round(macro_f1, 4)},
        ]
    )
    return eval_df, overall, intent_results, by_intent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare model citation-intent predictions with expert human "
            "labels in human_evaluation_Tuan_Anh_Phan.zip."
        )
    )
    parser.add_argument(
        "--zip-path",
        required=True,
        type=Path,
        help="Path to human_evaluation_Tuan_Anh_Phan.zip (Zenodo Version 3).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs (defaults to <zip-dir>/human_evaluation_results).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path
    if not zip_path.exists():
        raise FileNotFoundError(f"Evaluation archive not found: {zip_path}")

    out_dir = args.out_dir or (zip_path.parent / "human_evaluation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = load_evaluation_archive(zip_path)
    eval_df, overall, intent_results, by_intent = evaluate(combined)

    results = {
        "human_evaluation_overall.csv": overall,
        "human_evaluation_per_intent.csv": intent_results,
        "human_evaluation_agreement_by_intent.csv": by_intent,
    }
    for fname, df in results.items():
        out_path = out_dir / fname
        df.to_csv(out_path, index=False)
        print(f"\n=== {fname} ===")
        print(df.to_string(index=False))

    eval_out = out_dir / "human_evaluation_records.csv"
    eval_df.drop(columns=["model_set", "human_set"]).to_csv(eval_out, index=False)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
