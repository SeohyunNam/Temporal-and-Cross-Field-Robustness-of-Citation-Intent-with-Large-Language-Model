# Pipeline 3 — EdgeCite Construction

Reorganizes the **Citation context & intent data** into **EdgeCite**, a
retrieval-oriented, citation-event-level benchmark for leakage-controlled,
intent-conditioned citation retrieval.

## Flow

```
Citation context & intent data  ─┐
(citing_contexts.json)           │  build_edgecite.py
Top 5% cited papers per journal ─┘
(*_for_paper_title.csv anchors)
        │  1. contexts -> edges (citing_doi, cited_doi, field, context, primary_label)
        │  2. keep canonical primary intent + non-empty context
        │  3. dedup + boilerplate/noise cleaning
        │  4. join cited-anchor title (title-based candidates)
        │  5. citing-disjoint train/validation/test split (80/10/10)
        │  6. attach citing-paper year
        ▼
retrieval_dataset_citing_disjoint_with_year.parquet
```

## Output schema

| column | description |
| --- | --- |
| `citing_doi` | DOI of the citing paper |
| `cited_doi_norm` | normalized DOI of the cited anchor paper |
| `cited_title` | title of the cited anchor paper |
| `field` | broad scientific field |
| `context` | local citation context text |
| `primary_label` | primary (canonical) citation intent |
| `split` | `train` / `validation` / `test` (citing-disjoint) |
| `year` | citing-paper publication year |

The **citing-disjoint** split guarantees that no citing paper appears in more
than one split, controlling for document-level leakage in retrieval evaluation.
Candidate anchors are represented by their titles, and only citation events
whose cited anchor has a valid title are retained.

## Usage

```bash
python build_edgecite.py \
    --context-zip "Citation context & intent data.zip" \
    --top5-zip "Top 5% cited papers per journal dataset.zip" \
    --out retrieval_dataset_citing_disjoint_with_year.parquet
```

`--context-dir` / `--top5-dir` can be used instead of the ZIP arguments when
the inputs are already extracted. Dependencies: see
[`../requirements.txt`](../requirements.txt).
