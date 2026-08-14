# Pipeline 1 — Data Collection & Preprocessing

Builds the **Citation context & intent data** from scratch: from raw Scopus
records all the way to the per–seed-paper `citing_contexts.json` artifacts that
every later pipeline (MDCite, EdgeCite, IDCite) consumes.

## Flow

```
Scopus (Year 2000-2024)                      collect_by_journal.py
        │                                    (Scopus API, per journal)
        ▼
Top-5 Q1 journals per field                  build_field_journal_mapping.py
(WoS / JCR categories)                       (match journals ↔ Scopus CSVs)
        │
        ▼
per-journal full datasets                    extract_journal_full_datasets.py
        │
        ▼
Top 5% cited papers per journal              select_top5pct_per_journal.py
(seed papers)                                (journal-stratified top 5%)
        │
        ▼
Citation context extraction                  batch_paper_title_multi.py
+ citation intent classification             (+ paper_title.py engine)
        │
        ▼
Citation context & intent data               output_<field>/<doi>/citing_contexts.json
```

## Scripts

| Script | Role | Main output |
| --- | --- | --- |
| `collect_by_journal.py` | Collect Scopus records for one journal / year range | `scopus_<JOURNAL>_2000_2024.csv` |
| `build_field_journal_mapping.py` | Match journals to WoS field/category groups | merged field/journal CSV |
| `extract_journal_full_datasets.py` | Split merged table into per-journal files | `<group>__<journal>_full.csv` |
| `select_top5pct_per_journal.py` | Journal-stratified top-5% seed selection | `<group>_top5pct_per_journal_for_paper_title.csv` |
| `paper_title.py` | Citation-context extraction engine (OpenAlex + Semantic Scholar) | per-DOI `citing_contexts.json` |
| `batch_paper_title_multi.py` | Batch driver over all seed CSVs | `output_<field>/<doi>/citing_contexts.json` |

## Citation intent classification

Citation **contexts** are retrieved via the Semantic Scholar Graph API. The
citation **intent** for each retrieved context is then produced by running the
SynIntent model — the generative, heterogeneous-graph-neural-network intent
model of Phan et al. (*Understanding citation intents by generative intent
model based on heterogeneous graph neural network*, Information Processing &
Management, 2026, 63(6):104743). The resulting intent-annotated records are
stored in `citing_contexts.json`.

> The SynIntent model and its training code belong to that external work and
> are **not** redistributed in this repository. This pipeline covers context
> retrieval and the surrounding orchestration; the intent-classification
> substep is a call-out to the external SynIntent model.

## Requirements & credentials

Install dependencies from [`../requirements.txt`](../requirements.txt). API
keys are read from environment variables and are never stored in the repo:

```bash
export SCOPUS_APIKEY=...           # Scopus (institutional entitlement may apply)
export SCOPUS_INSTTOKEN=...        # optional
export SEMANTIC_SCHOLAR_API_KEY=... # optional (raises rate limits)
```

## Example

```bash
# 1. Collect Scopus records per journal (repeat per journal/ISSN)
python collect_by_journal.py --issn 0140-6736 --year-from 2000 --year-to 2024 --out lancet.csv

# 2. Map journals to WoS field/category groups
python build_field_journal_mapping.py \
    --scopus-zip "Scopus (Year 2000 - 2024).zip" \
    --group-file wos_groups.txt \
    --out WOS_merged_preprocessed.csv

# 3. Split into per-journal full datasets
python extract_journal_full_datasets.py \
    --merged-csv WOS_merged_preprocessed.csv \
    --out-dir Field_full_datasets_by_journal

# 4. Journal-stratified top-5% seed selection
python select_top5pct_per_journal.py \
    --journal-full-dir Field_full_datasets_by_journal \
    --out-dir Field_top5pct_per_journal

# 5. Extract citation contexts + intents for every seed paper
python batch_paper_title_multi.py --mode multi \
    --csv-dir Field_top5pct_per_journal \
    --out-base-dir citation_context_intent_data
```
