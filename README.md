<p align="center">
  <img src="assets/bannerwide.png" alt="QueryBooks Banner" width="100%">
</p>

# QueryBooks rag-application

Two dedicated apps: a loader that chunks/passes PDFs into knowledge base (hybrid postgres) and a chat/query surface that retrieves context from the populated tables, and answers user's questions based on the source material.

## Web UIs

* **Loader (`apps/loader/app.py`)** – Bring up the loader container with `docker compose --profile ui up loader`. The Gradio UI still mirrors `apps/loader/pipeline.py`’s options, but the containerized loader now only runs inside Docker and admits browser connections on `localhost:7860`.
* **Chat (`apps/query/app.py`)** – Start the chat UI with `docker compose --profile ui up query` (or boot both UIs together via `docker compose --profile ui up loader query`). It defaults to the `md` profile but reads all the same `DEPLOY_*` values that the loader writes so the chat responses always match the latest data.

All three target tables can coexist—choose `mini`, `sm`, or `md` depending on your experimentation goals. You can populate just one table or fill them all, and the only shared config needed is the `MEDIUM_PROVIDER`/`DEPLOY_*` set of env vars for medium embeddings.

Both UIs and the CLI share the same env-loading order: `project.env`, `.env`, then `--env-file` (pipeline default is `.env`).

To avoid re-downloading Hugging Face models every time, set the `HF_CACHE_PATH` environment variable in `.env` (for example `/home/<you>/.cache/huggingface` on Linux/macOS or `C:/Users/<you>/.cache/huggingface` on Windows). Compose mounts that path into each container at `/root/.cache/huggingface`, ensuring the same cache serves both services and any host process that shares the directory. Start the UIs with `docker compose --profile ui up loader query` and open the loader (`localhost:7860`) or query (`localhost:7861`) in your browser.

## Data population

```bash
docker compose run --rm loader python -m apps.loader.pipeline
```

That command truncates `knowledge_base_mini`, `knowledge_base_sm`, and `knowledge_base_md` (unless you pass `--skip-empty`) and populates the mini/small tables by default. The medium step honors `MEDIUM_PROVIDER` (default `openai`), so you only need Azure creds if you explicitly run `--medium-provider azure`.

## Containerized apps

`compose.yaml` now includes dedicated loader and query services that build from `Dockerfile.loader` and `Dockerfile.query` (they install dependencies via `uv sync` and launch the loader/query entrypoints from the repository). They mount the repo so the UI code and env files stay in sync with your workspace and share the local Postgres service, so you can keep using `docker compose` for the database without change. The loader/query containers belong to the `ui` profile, so they only start when you explicitly request that profile (e.g., `docker compose --profile ui up loader query`). The profile declaration keeps the database/migrations stack unaffected while still allowing you to bring up both UIs together whenever you need them.

To avoid re-downloading Hugging Face models every time, set the `HF_CACHE_PATH` environment variable in `.env` (for example `/home/<you>/.cache/huggingface` on Linux/macOS or `C:/Users/<you>/.cache/huggingface` on Windows). Compose mounts that path into each container at `/root/.cache/huggingface`, ensuring the same cache serves both services and any host process that shares the directory. Start the UIs with `docker compose up loader query` and open the loader (`localhost:7860`) or query (`localhost:7861`) in your browser.

## Prerequisites

Start the local DB/migrations:

```bash
docker compose up -d database
docker compose run --rm database-migrations
```

The loader/query Dockerfiles install Python dependencies via `uv sync` during `docker compose build`. Rebuild those services only after changing `pyproject.toml` or `uv.lock`.

## Technical stack

- **Docker Compose** – orchestrates the `pgvector_database` (based on the PostgreSQL 18/pgvector image) and `database-migrations` containers that run Flyway against the shared volume. The base table schema, pgvector vectors, and pg_search indexes are defined via the `database-migrations` service.
- **PDF + chunking pipeline** – `scripts/preprocess.py` relies on `pdfplumber`, spaCy (`en_core_web_sm` default), and sentence-transformers/OpenAI/Azure embeddings for chunking and encoding before pushing rows into Postgres.
- **LangChain + Hugging Face** – `chat_manager.py` builds retrievers (semantic/lexical/hybrid via PGVector/Hugging Face + pg_search) plus reranking retrievers that call `gpt-4o-mini`. Hugging Face tokenizers/models run locally (`sentence-transformers`), while other refreshing embeddings may hit Azure or OpenAI based on `DEPLOY_*` env vars.
- **Gradio / CLI entry points** – the loader uses Gradio (`apps/loader/app.py`) to wrap the same preprocess arguments, while the CLI script in `apps/loader/pipeline.py` provides shortcuts for truncation, preprocessing, and step selection. Both share `utils/loader_utils.py` and the `scripts/preprocess.py` invocation.
- **Supporting libraries** – `psycopg2`/`pgvector`/`pg_search` connectors, `dotenv` env loading, and `uvicorn`/`uv` task runner glue everything together under the `pyproject.toml` dependencies so the repo stays interpreter-agnostic.

## CLI options

```bash
docker compose run --rm loader python -m apps.loader.pipeline --help
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
PGHOST=database
PGPORT=5432
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=password
PGSSLMODE=disable
```

### Chunking and preprocessing defaults

Embedding model is needed for semantic chunking. The loader UI container exposes only the local provider so the options below describe that path; Azure chunking/deployments are available only when you run the CLI inside the loader container (for example `docker compose run --rm loader python -m apps.loader.pipeline --medium-provider azure` or `docker compose run --rm loader python scripts/preprocess.py ...`) and explicitly request `--medium-provider azure` (or set `MEDIUM_PROVIDER=azure`).

```env
PDF_PATH=data/euaiact.pdf
SOURCE_NAME=euaiact.pdf

CHUNKER=spacy
SPACY_MODEL=en_core_web_sm
MAX_SENTENCES=5

CHUNKING_PROVIDER=local
CHUNKING_LOCAL_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNKING_DEPLOYMENT= # only used when running the CLI with `CHUNKING_PROVIDER=azure`
BREAKPOINT_THRESHOLD_TYPE=percentile

MAX_EMBED_TOKENS=2000
SPLIT_OVERLAP_TOKENS=80
```

### spaCy models

The loader relies on spaCy models being installed in its container virtualenv. The default `en-core-web-sm` wheel is pinned in `pyproject.toml`, so it becomes available after the loader container is built. To add another spaCy model, append the wheel URL to `pyproject.toml` (for example `en-core-web-trf @ https://...`) and rebuild the loader service (`docker compose build loader`). As a shortcut you can install a model temporarily with `docker compose run --rm loader python -m pip install <name>` or `docker compose run --rm loader python -m spacy download <name>`, but updating `pyproject.toml` keeps dependency management consistent.

### Defaul openai endpoint

For openai endpoints (medium profiles default behavious), you need to provide the api key:
```env
OPENAI_API_KEY=sk-<your-openai-key>
```

### OpenAI model overrides

Use these to choose which OpenAI chat/ reranking models power the query UI:
```env
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_RERANK_MODEL=gpt-4o-mini
```

### Azure embedding overrides *(CLI only)*

The Gradio loader/query UIs always default to OpenAI for the medium profile. Azure embeddings are still supported when you run the CLI inside the loader container (for example `docker compose run --rm loader python -m apps.loader.pipeline --medium-provider azure` or `docker compose run --rm loader python scripts/preprocess.py ...`) and explicitly pass `--medium-provider azure` (or set `MEDIUM_PROVIDER=azure` in the `.env`). That flow also requires the Azure endpoint/deployment values below; use it only if you know how to configure your Azure deployments and Postgres endpoint.

```env
AZURE_ENDPOINT=...
AZURE_API_KEY=...
DEPLOY_MEDIUM=...
AZURE_API_VERSION=2025-03-01-preview
```

**NOTE:** The use of Azure embeddings is discouraged at the moment. It is still possible to use Azure embeddings if you, for example, want to use Azure Flex server database and bulk load it using `docker compose run --rm loader python -m apps.loader.pipeline`. However, the query end can only use either local or OpenAI embedding models. Adding full support for Foundry deployments is not at the top of the todo list, but might happen at a later date.

### Embedding profile hints

Set `PROFILE` (`mini`, `sm`, or `md`) to tell `apps/query/app.py` which table/embedding pair to use. The chat manager then pulls the embedding model from the matching `DEPLOY_*` env var, so swap in gated models as needed.

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


## Notes

Both the loader and the query UI already run inside their own Docker containers, so the workspace now relies entirely on `docker compose` for startup and development. If you need to inspect either container, use `docker compose ps` and `docker compose logs --follow <service>` as usual.

### Gradio warning

Gradio currently logs a warning about the `chatbot` component still emitting the old tuple-based format (`type='tuples'`). You can silence it by switching `apps/query/app.py` to `Chatbot(type='messages')` or waiting for a Gradio release that drops the old format; nothing in this repo needs to change until that happens, but updating the component to use `type='messages'` keeps the UI warning-free when you next refresh the dependencies.
