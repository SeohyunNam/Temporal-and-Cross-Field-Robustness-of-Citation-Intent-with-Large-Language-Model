# Pipeline 4 — IDCite Construction

Builds the ontology-ready **IDCite** release: 17 Parquet tables covering
citation events, publications, normalized scholarly entities, and a
supplementary heterogeneous knowledge graph.

## Flow

```
Top 5% cited papers per journal (seed papers)  ─┐
Citation context & intent data                  ┘  ontology.py
        │  citation-event construction
        │  entity normalization (authors, affiliations, journals,
        │  cities, countries, fields, intents)
        │  ontology-ready knowledge-graph assembly
        ▼
17 Parquet tables (citation events, publications, entities, kg_nodes/kg_edges)
```

## Inputs

`ontology.py` expects, under `--base-dir`:

```
<base-dir>/Top 5% cited papers per journal dataset.zip   (seed *_full.json)
<base-dir>/data_1242025_result_revised.zip               (citing_contexts.json)
```

## Outputs (`<base-dir>/citationhub_v1_ontology_ready/`)

Citation events: `citation_events.parquet`, `citation_events_enriched.parquet`,
`citation_events_normalized.parquet`
Publications: `citing_papers.parquet`, `citing_papers_normalized.parquet`,
`seed_cited_papers.parquet`, `seed_cited_papers_normalized.parquet`
Normalized entities: `authors.parquet`, `affiliations.parquet`,
`affiliation_geo.parquet`, `journals.parquet`, `fields.parquet`,
`intents.parquet`, `cities.parquet`, `countries.parquet`
Knowledge graph: `kg_nodes.parquet`, `kg_edges.parquet`

## Usage

```bash
python ontology.py --base-dir /path/to/wos_data
```

Technical validation of these tables is provided separately in
[`../Technical Validation/`](../Technical%20Validation/). Dependencies: see
[`../requirements.txt`](../requirements.txt).
