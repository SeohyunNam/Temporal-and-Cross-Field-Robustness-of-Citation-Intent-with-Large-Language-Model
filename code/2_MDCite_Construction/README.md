# Pipeline 2 — MDCite Construction

Turns the **Citation context & intent data** (Pipeline 1 output) into the
released **MDCite** citation-context dataset.

## Flow

```
Citation context & intent data              build_mdcite.py
(output_<field>/<doi>/citing_contexts.json)
        │  parse contexts + intents
        │  keep single (non-composite) intent
        ▼
dataset_context_intent_single.parquet / .csv
dataset_context_intent.parquet / .csv        (multi-intent, provenance)
```

## Output schema

Each row is one citation context:

| column | description |
| --- | --- |
| `text` | citation context text |
| `label` | functional citation intent |
| `field` | scientific field (`output_<field>`) |
| `group_id` | citing-paper identifier (document-level split key) |
| `paperId` | Semantic Scholar work identifier |
| `doi` | DOI |
| `venue` | publication venue |
| `year` | publication year |
| `source_file` | provenance identifier |

The **single-intent** variant keeps only rows whose label is a single canonical
intent (composite labels containing a space are dropped), giving the released
`dataset_context_intent_single.{parquet,csv}` benchmark. The multi-intent table
is retained for provenance.

## Usage

```bash
# Directly from the ZIP archive
python build_mdcite.py \
    --context-zip "Citation context & intent data.zip" \
    --out-dir mdcite_out

# Or from an extracted directory
python build_mdcite.py \
    --context-dir citation_context_intent_data \
    --out-dir mdcite_out
```

Dependencies: see [`../requirements.txt`](../requirements.txt).
