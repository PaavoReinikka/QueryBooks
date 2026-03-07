# Tests

This folder contains integration tests for hybrid search SQL migrations.

## What is covered

- `content_tsv` columns exist on all knowledge base tables
- Hybrid SQL functions exist:
  - `kb_rrf_score`
  - `search_knowledge_base_mini_hybrid`
  - `search_knowledge_base_sm_hybrid`
  - `search_knowledge_base_md_hybrid`
- Hybrid functions return rows and are sorted by descending `rrf_score`

## Prerequisites

- Database container is running (`docker compose up -d database`)
- Migrations are applied (`docker compose run --rm database-migrations`)
- Data has been populated (e.g. `docker compose run --rm loader python -m apps.loader.pipeline` or via the loader UI container)
- Env vars for DB are available in `.env` (or `project.env`) with `PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD`

## Run

```bash
python -m unittest -v tests/test_hybrid_search_db.py
```
