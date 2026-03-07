from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_SCRIPT = ROOT / "scripts" / "preprocess.py"


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def load_environment(env_file: str) -> None:
    load_dotenv(ROOT / "project.env", override=False)
    load_dotenv(ROOT / ".env", override=True)
    load_dotenv(ROOT / env_file, override=True)


def build_preprocess_args(loader_args: SimpleNamespace) -> List[str]:
    pdf_path = loader_args.pdf_path
    source = loader_args.source or Path(pdf_path).name

    args: List[str] = [
        "--pdf-path",
        pdf_path,
        "--source",
        source,
        "--spacy-model",
        loader_args.spacy_model or env_or_default("SPACY_MODEL", "en_core_web_sm"),
        "--max-sentences",
        str(loader_args.max_sentences or env_or_default("MAX_SENTENCES", "5")),
        "--chunker",
        loader_args.chunker or env_or_default("CHUNKER", "spacy"),
        "--chunking-provider",
        loader_args.chunking_provider or env_or_default("CHUNKING_PROVIDER", "local"),
        "--chunking-local-model",
        loader_args.chunking_local_model
        or env_or_default("CHUNKING_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "--chunking-deployment",
        loader_args.chunking_deployment
        or os.getenv("CHUNKING_DEPLOYMENT")
        or os.getenv("DEPLOY_MEDIUM")
        or "",
        "--breakpoint-threshold-type",
        loader_args.breakpoint_threshold
        or env_or_default("BREAKPOINT_THRESHOLD_TYPE", "percentile"),
        "--max-embed-tokens",
        str(loader_args.max_embed_tokens or env_or_default("MAX_EMBED_TOKENS", "2000")),
        "--split-overlap-tokens",
        str(loader_args.split_overlap_tokens or env_or_default("SPLIT_OVERLAP_TOKENS", "80")),
        "--pg-host",
        env_or_default("PGHOST", "localhost"),
        "--pg-port",
        env_or_default("PGPORT", "5431"),
        "--pg-database",
        env_or_default("PGDATABASE", "postgres"),
        "--pg-user",
        env_or_default("PGUSER", "postgres"),
        "--pg-password",
        env_or_default("PGPASSWORD", "password"),
    ]

    if loader_args.azure_deployment:
        args.extend(["--deployment", loader_args.azure_deployment])
    elif os.getenv("DEPLOY_MEDIUM"):
        args.extend(["--deployment", os.getenv("DEPLOY_MEDIUM")])

    if loader_args.azure_endpoint:
        args.extend(["--azure-endpoint", loader_args.azure_endpoint])
    elif os.getenv("AZURE_ENDPOINT"):
        args.extend(["--azure-endpoint", os.getenv("AZURE_ENDPOINT")])

    if loader_args.azure_api_key:
        args.extend(["--azure-api-key", loader_args.azure_api_key])
    elif os.getenv("AZURE_API_KEY"):
        args.extend(["--azure-api-key", os.getenv("AZURE_API_KEY")])

    if loader_args.dry_run:
        args.append("--dry-run")

    return args


def run_step(*, step_args: Dict[str, List[str]], base_args: List[str]) -> None:
    cmd = [sys.executable, str(PREPROCESS_SCRIPT), *base_args, *step_args["args"]]
    print(f"> Running step: {step_args['name']}" )
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Step '{step_args['name']}' failed with exit code {result.returncode}: {joined}"
        )


def mini_step() -> Dict[str, List[str]]:
    return _local_step(
        table_name="knowledge_base_mini",
        model_env="DEPLOY_MINI",
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    )


def small_step() -> Dict[str, List[str]]:
    return _local_step(
        table_name="knowledge_base_sm",
        model_env="DEPLOY_SMALL",
        default_model="BAAI/bge-large-en-v1.5",
    )


def _local_step(table_name: str, model_env: str, default_model: str) -> Dict[str, List[str]]:
    return {
        "name": table_name,
        "args": [
            "--table",
            table_name,
            "--provider",
            "local",
            "--local-model",
            env_or_default(model_env, default_model),
        ],
    }


def medium_step(loader_args: SimpleNamespace) -> Dict[str, List[str]]:
    provider = (
        loader_args.medium_provider
        or os.getenv("MEDIUM_PROVIDER")
        or "openai"
    ).lower()
    if provider not in ("openai", "azure"):
        raise ValueError("medium provider must be either 'openai' or 'azure'")

    deployment = loader_args.azure_deployment or os.getenv("DEPLOY_MEDIUM")
    if not deployment:
        raise ValueError(
            "Cannot run medium step without a deployment/model. Set DEPLOY_MEDIUM in .env or pass --azure-deployment."
        )

    args = [
        "--table",
        "knowledge_base_md",
        "--provider",
        provider,
        "--deployment",
        deployment,
    ]

    if provider == "azure":
        endpoint = loader_args.azure_endpoint or os.getenv("AZURE_ENDPOINT")
        api_key = loader_args.azure_api_key or os.getenv("AZURE_API_KEY")
        missing = [
            name
            for name, value in {
                "AZURE_ENDPOINT": endpoint,
                "AZURE_API_KEY": api_key,
            }.items()
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Cannot run Azure md step; missing env vars: {joined}. "
                "Set them in .env or pass them explicitly when requesting the Azure provider."
            )

        args.extend([
            "--azure-endpoint",
            endpoint,
            "--azure-api-key",
            api_key,
        ])

    if provider == "openai" and not (loader_args.openai_api_key or os.getenv("OPENAI_API_KEY")):
        raise ValueError(
            "OpenAI medium step requires OPENAI_API_KEY in the environment or via --openai-api-key."
        )

    return {"name": "knowledge_base_md", "args": args}


def select_step(loader_args: SimpleNamespace) -> Dict[str, List[str]]:
    if loader_args.mini:
        return mini_step()
    if loader_args.small:
        return small_step()
    if loader_args.medium:
        return medium_step(loader_args)
    raise ValueError("Select a target table before loading.")