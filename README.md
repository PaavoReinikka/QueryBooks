<p align="center">
  <img src="assets/bannerwide.png" alt="QueryBooks Banner" width="100%">
</p>

# QueryBooks rag-application

Two dedicated apps: a loader that chunks/passes PDFs into knowledge base (hybrid postgres) and a chat/query surface that retrieves context from the populated tables, and answers user's questions based on the source material.

## Web UIs

* **Loader (`app_loader.py`)** – Gradio UI mirroring `pipeline.py`’s options. Upload or point to a PDF, select mini/small/medium, tweak chunking/provider overrides, and truncate/dry-run before writing. Launch with `uv run app_loader.py`; set `PROFILE` before launching if you want a different default target table (`md` is the default profile, so the loader starts on medium unless you override it). The medium step now defaults to OpenAI embeddings (`MEDIUM_PROVIDER=openai`), which means the loader expects `OPENAI_API_KEY` to be set; switching to Azure requires explicitly changing the dropdown or passing `--medium-provider azure` along with the Azure endpoint/key.
* **Chat (`app_query.py`)** – Tool-enabled assistant that queries the populated tables. It checks `PROFILE` (`mini`, `sm`, or `md`, default `md`) to decide which table/embedding pair to hit and reads the corresponding embedding model from `DEPLOY_MINI`, `DEPLOY_SMALL`, or `DEPLOY_MEDIUM`. The medium profile still uses whatever model is configured in `DEPLOY_MEDIUM`, so it will match the loader’s provider configuration (OpenAI by default). Run `uv run app_query.py` after loading data. To launch the UI with a specific profile, prefix the command with `PROFILE=mini uv run ...` (or `sm`/`md`).

Both UIs and the CLI share the same env-loading order: `project.env`, `.env`, then `--env-file` (pipeline default is `.env`).

## Local data population (single command)

```bash
uv run pipeline.py
```

That run truncates `knowledge_base_mini`, `knowledge_base_sm`, and `knowledge_base_md` (unless you set `--skip-empty`) and populates the mini/small tables by default. The medium step no longer insists on Azure; it honors `MEDIUM_PROVIDER` (default `openai`) so you only need Azure creds if you select `azure` explicitly.

## Prerequisites

Start the local DB/migrations:

```bash
docker compose up -d database
docker compose run --rm database-migrations
```

Install dependencies (one-time):

```bash
uv sync
```

## Technical stack

- **Docker Compose** – orchestrates the `pgvector_database` (based on the PostgreSQL 18/pgvector image) and `database-migrations` containers that run Flyway against the shared volume. The base table schema, pgvector vectors, and pg_search indexes are defined via the `database-migrations` service.
- **PDF + chunking pipeline** – `scripts/preprocess.py` relies on `pdfplumber`, spaCy (`en_core_web_sm` default), and sentence-transformers/OpenAI/Azure embeddings for chunking and encoding before pushing rows into Postgres.
- **LangChain + Hugging Face** – `chat_manager.py` builds retrievers (semantic/lexical/hybrid via PGVector/Hugging Face + pg_search) plus reranking retrievers that call `gpt-4o-mini`. Hugging Face tokenizers/models run locally (`sentence-transformers`), while other refreshing embeddings may hit Azure or OpenAI based on `DEPLOY_*` env vars.
- **Gradio / CLI entry points** – the loader uses Gradio (`app_loader.py`) to wrap the same preprocess arguments, while `pipeline.py` provides a CLI shortcut to truncation, preprocessing, and step selection. Both share `utils/loader_utils.py` and the `scripts/preprocess.py` invocation.
- **Supporting libraries** – `psycopg2`/`pgvector`/`pg_search` connectors, `dotenv` env loading, and `uvicorn`/`uv` task runner glue everything together under the `pyproject.toml` dependencies so the repo stays interpreter-agnostic.

## CLI options

```bash
uv run pipeline.py --help
```

### Table targeting

- `--mini`: populate only `knowledge_base_mini`
- `--small`: populate only `knowledge_base_sm`
- `--medium`: populate only `knowledge_base_md` (it defaults to OpenAI embeddings; set `MEDIUM_PROVIDER=azure` or pass `--medium-provider azure` if you need Azure)
- `--all`: run mini + small + medium
- `--skip-empty`: skip the initial truncation step
- `--dry-run`: preprocess without writing

### Input overrides

- `--pdf-path`: override the PDF path
- `--source`: override the source name stored in the DB
- `--env-file`: extra env file loaded last (defaults to `.env`)

## Environment variables

All runners load env vars from `project.env`, then `.env`, then `--env-file` (if supplied).

### Local Postgres

```env
PGHOST=localhost
PGPORT=5431
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=password
PGSSLMODE=disable
```

### Chunking and preprocessing defaults

Embedding model is needed for semantic chunking. Small local (HF) models are usually sufficient for this. If for some reason you want to use API's, only azure endpoints are supported at the moment -- you need to modify the `preprocess.py` script yourself for other options -- yes, this is inconsistant with the way the query end is working (defaulting to openai for API's). This might change in the future to support opeanai as the primary API.

```env
PDF_PATH=data/euaiact.pdf
SOURCE_NAME=euaiact.pdf

CHUNKER=spacy
SPACY_MODEL=en_core_web_sm
MAX_SENTENCES=5

CHUNKING_PROVIDER=local
CHUNKING_LOCAL_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNKING_DEPLOYMENT= #only if provider azure 
BREAKPOINT_THRESHOLD_TYPE=percentile

MAX_EMBED_TOKENS=2000
SPLIT_OVERLAP_TOKENS=80
```

### Defaul openai endpoint

For openai endpoints (medium profiles default behavious), you need to provide the api key:
```env
OPENAI_API_KEY=sk-<your-openai-key>
```

### Azure embedding overrides *(optional for medium when `MEDIUM_PROVIDER=azure`)*

If you have access to azure api's (e.g., foundry deployments), you can specify that you wish to use azure by passing the provider when launcing the apps, or by setting it in the env. Then you need to also set the following vars:

```env
AZURE_ENDPOINT=...
AZURE_API_KEY=...
DEPLOY_MEDIUM=...
AZURE_API_VERSION=2025-03-01-preview
```

### Embedding profile hints

Set `PROFILE` (`mini`, `sm`, or `md`) to tell `app_query.py` which table/embedding pair to use. The chat manager then pulls the embedding model from the matching `DEPLOY_*` env var, so swap in gated models as needed.

### Hugging Face

Provide `HF_TOKEN` to avoid Hugging Face rate-limit warnings, especially when using gated medium embeddings:

```env
HF_TOKEN=...
```

### Chat memory limit

Control how many tokens stay in memory during a conversation with `MAX_MEMORY_TOKENS` (defaults to `8000`). Setting it lower trims older messages more aggressively, which can be useful when you need to keep the chat state smaller for latency or cost reasons:

```env
MAX_MEMORY_TOKENS=6000
```


## (Near) Future changes

At the moment, only the database is running in a container. This will change in the near future, and both the loader and the query ends will be running in a dedicated containers. This simplifies both local use and possible cloud deployments.
