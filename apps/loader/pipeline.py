from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Dict, List

import psycopg2

from utils.loader_utils import (
    build_preprocess_args,
    load_environment,
    medium_step,
    mini_step,
    run_step,
    small_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform local population pipeline (no .sh/.ps1 required)."
    )
    parser.add_argument("--env-file", default=".env", help="Env file to load last (default: .env)")
    parser.add_argument("--pdf-path", default=None, help="PDF path override")
    parser.add_argument("--source", default=None, help="Source name override")
    parser.add_argument("--skip-empty", action="store_true", help="Skip truncating target tables")
    parser.add_argument("--dry-run", action="store_true", help="Run preprocess in dry-run mode")
    parser.add_argument("--mini", action="store_true", help="Populate only `knowledge_base_mini`")
    parser.add_argument("--small", action="store_true", help="Populate only `knowledge_base_sm`")
    parser.add_argument("--medium", action="store_true", help="Populate only `knowledge_base_md`")
    parser.add_argument("--all", action="store_true", help="Populate mini + small + medium tables")
    parser.add_argument(
        "--medium-provider",
        choices=["openai", "azure"],
        default=None,
        help="Medium embedding provider (defaults to MEDIUM_PROVIDER or openai)",
    )
    return parser.parse_args()


def build_pg_dsn() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5431")
    database = os.getenv("PGDATABASE", "postgres")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "password")
    sslmode = os.getenv("PGSSLMODE", "disable")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


def truncate_tables() -> None:
    dsn = build_pg_dsn()
    sql = "TRUNCATE TABLE knowledge_base_mini, knowledge_base_sm, knowledge_base_md RESTART IDENTITY;"
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Tables truncated successfully.")


def _build_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    pdf_path = args.pdf_path or os.getenv("PDF_PATH") or "data/euaiact.pdf"
    return SimpleNamespace(
        env_file=args.env_file,
        pdf_path=pdf_path,
        source=args.source,
        chunker=None,
        chunking_provider=None,
        chunking_local_model=None,
        chunking_deployment=None,
        max_sentences=None,
        spacy_model=None,
        breakpoint_threshold=None,
        max_embed_tokens=None,
        split_overlap_tokens=None,
        azure_deployment=None,
        azure_endpoint=None,
        azure_api_key=None,
        medium_provider=args.medium_provider,
        openai_api_key=None,
        mini=args.mini,
        small=args.small,
        medium=args.medium,
        truncate=not args.skip_empty,
        dry_run=args.dry_run,
    )


def _build_steps(args: argparse.Namespace, loader_args: SimpleNamespace) -> List[Dict[str, List[str]]]:
    if args.all and (args.mini or args.small or args.medium):
        raise ValueError("`--all` cannot be combined with `--mini`, `--small`, or `--medium`.")

    selected: List[Dict[str, List[str]]] = []
    if args.all:
        selected.extend([mini_step(), small_step(), medium_step(loader_args)])
        return selected

    any_specific = args.mini or args.small or args.medium
    if any_specific:
        if args.mini:
            selected.append(mini_step())
        if args.small:
            selected.append(small_step())
        if args.medium:
            selected.append(medium_step(loader_args))
        return selected

    return [mini_step(), small_step()]


def main() -> None:
    args = parse_args()
    load_environment(args.env_file)

    if not args.skip_empty and not args.dry_run:
        truncate_tables()
    elif args.dry_run:
        print("Dry run enabled; skipping table truncation.")

    loader_args = _build_loader_args(args)
    base_args = build_preprocess_args(loader_args)

    steps = _build_steps(args, loader_args)
    for step in steps:
        run_step(step_args=step, base_args=base_args)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
