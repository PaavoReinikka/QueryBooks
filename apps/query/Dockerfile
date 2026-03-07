FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_VENV=/opt/.venv \
    PATH=/opt/.venv/bin:$PATH

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock project.env ./

RUN pip install --no-cache-dir uv \
    && uv sync \
    && mv .venv /opt/.venv

COPY . .

CMD ["uv", "run", "-m", "apps.query.app"]
