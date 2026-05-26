# Task Summary: Seed PM Cards Notebook — Bronze/Gold Layers + Vector Search

## Deliverable

**Notebook:** `anomaly-detection/notebooks/Seeding-PM-Cards-Bronze-Gold.ipynb`

## What was built

A Databricks-ready notebook (21 cells, Python + `spark.sql()` pattern matching existing notebooks) that:

### Phase 1 — Bronze Layer
- Creates schema `demo.pm_cards_bronze` and Delta table `pm_cards` with 6 raw columns
- Seeds all 45 PM card rows (R-00: 11, R-01: 10, R-02: 10, R-03: 10, shared fleet: 5) via `DELETE FROM` + `INSERT` for idempotency
- Verifies row count = 45 with group-by breakdown

### Phase 2 — Gold Layer
- Creates schema `demo.pm_cards_gold` and Delta table `pm_cards` with 8 columns (6 raw + `combined_text` + `embedding ARRAY<FLOAT>`)
- Delta table has Change Data Feed enabled (required for Vector Search Delta Sync)
- Reads bronze, enriches with `combined_text` via `concat_ws(". ", cause, symptom, action)`
- Pre-computes 384-dim embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Writes gold table with overwriteSchema + verifies row count and embedding dimension

### Phase 3 — Vector Search
- Creates/verifies Vector Search endpoint (`pm-cards-vs-endpoint`, STANDARD type)
- Creates Delta Sync Vector Index on `demo.pm_cards_gold.pm_cards_vs_index` using `combined_text` with `databricks-bge-large-en` model endpoint
- Tests 3 semantic queries via `query_text` (server-side embedding computation) — wire feed speed jam, temperature rising, unstable arc

## Key design decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Schemas | `demo.pm_cards_bronze` / `demo.pm_cards_gold` | Purpose-named, distinct from flat `demo.robot_forensics` used by existing seed SQL |
| VS index type | Delta Sync with `embedding_source_columns` | `EmbeddingVectorColumn` not available in SDK v0.111.0; Databricks-managed embeddings are more maintainable |
| Query mode | `query_text` | Databricks VS endpoint computes query embeddings server-side, matching the index model |
| Cell format | Python + `spark.sql()` | Matches existing Training-V6 / Inference-V6 notebook patterns |
| Gold enrichment | Both `combined_text` + `embedding` | Embedding column available for ChromaDB/local fallback; VS index uses `combined_text` with model endpoint |

## Files changed

- **NEW**: `anomaly-detection/notebooks/Seeding-PM-Cards-Bronze-Gold.ipynb`

## Verification (to be run in Databricks)

1. Run all cells top-to-bottom — zero errors
2. `SELECT COUNT(*) FROM demo.pm_cards_bronze.pm_cards` → 45
3. `SELECT COUNT(*) FROM demo.pm_cards_gold.pm_cards` → 45
4. `SELECT combined_text, CARDINALITY(embedding) FROM demo.pm_cards_gold.pm_cards LIMIT 3` → non-null text, dim=384
5. Cell 20: 3/3 semantic queries return results with matching PM cards

## Out of scope

- Updating `backend/app/robot_agent/agent/rag_tools.py` to use Vector Search
- Silver layer (unnecessary for 45 static rows)
- CI/CD / scheduled refresh (PM cards are static reference data)
