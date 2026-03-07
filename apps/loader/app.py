import os
from types import SimpleNamespace

import gradio as gr

from utils.loader_utils import (
    build_preprocess_args,
    load_environment,
    run_step,
    select_step,
)
from utils.retrieval_functions import (
    connect_from_env,
    format_table_sources,
    inspect_table_sources,
)

load_environment(".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_CHUNKER = os.getenv("CHUNKER", "spacy")
DEFAULT_CHUNKING_PROVIDER = os.getenv("CHUNKING_PROVIDER", "local")
DEFAULT_CHUNKING_LOCAL_MODEL = os.getenv(
    "CHUNKING_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_CHUNKING_DEPLOYMENT = os.getenv("CHUNKING_DEPLOYMENT", "")
DEFAULT_BREAKPOINT = os.getenv("BREAKPOINT_THRESHOLD_TYPE", "percentile")
DEFAULT_MAX_SENTENCES = _int_env("MAX_SENTENCES", 5)
DEFAULT_SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
DEFAULT_MAX_EMBED_TOKENS = _int_env("MAX_EMBED_TOKENS", 2000)
DEFAULT_SPLIT_OVERLAP_TOKENS = _int_env("SPLIT_OVERLAP_TOKENS", 80)
DEFAULT_AZURE_DEPLOYMENT = os.getenv("DEPLOY_MEDIUM", "")
DEFAULT_AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
DEFAULT_AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
DEFAULT_MEDIUM_PROVIDER = os.getenv("MEDIUM_PROVIDER", "openai")


def _build_loader_args(
    pdf_path: str,
    source: str | None,
    table_choice: str,
    chunker: str | None,
    chunking_provider: str | None,
    chunking_local_model: str | None,
    chunking_deployment: str | None,
    max_sentences: int | None,
    spacy_model: str | None,
    breakpoint_threshold: str | None,
    max_embed_tokens: int | None,
    split_overlap_tokens: int | None,
    azure_deployment: str | None,
    azure_endpoint: str | None,
    azure_api_key: str | None,
    medium_provider: str | None,
    truncate: bool,
    dry_run: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        env_file=".env",
        pdf_path=pdf_path,
        source=source or None,
        chunker=chunker,
        chunking_provider=chunking_provider,
        chunking_local_model=chunking_local_model,
        chunking_deployment=chunking_deployment,
        max_sentences=max_sentences,
        spacy_model=spacy_model,
        breakpoint_threshold=breakpoint_threshold,
        max_embed_tokens=max_embed_tokens,
        split_overlap_tokens=split_overlap_tokens,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        medium_provider=medium_provider.lower() if medium_provider else None,
        openai_api_key=None,
        mini=table_choice == "mini",
        small=table_choice == "small",
        medium=table_choice == "medium",
        truncate=truncate,
        dry_run=dry_run,
    )


def run_loader(
    uploaded_file,
    pdf_path_input,
    source,
    table_choice,
    chunker,
    chunking_provider,
    chunking_local_model,
    chunking_deployment,
    max_sentences,
    spacy_model,
    breakpoint_threshold,
    max_embed_tokens,
    split_overlap_tokens,
    azure_deployment,
    azure_endpoint,
    azure_api_key,
    medium_provider,
    truncate,
    dry_run,
) -> str:
    pdf_path = uploaded_file or pdf_path_input or "data/silmarillion.pdf"
    if not pdf_path:
        return "Provide a PDF path or upload a file."

    loader_args = _build_loader_args(
        pdf_path=pdf_path,
        source=source or None,
        table_choice=table_choice,
        chunker=chunker if chunker else None,
        chunking_provider=chunking_provider if chunking_provider else None,
        chunking_local_model=chunking_local_model or None,
        chunking_deployment=chunking_deployment or None,
        max_sentences=int(max_sentences) if max_sentences else None,
        spacy_model=spacy_model or None,
        breakpoint_threshold=breakpoint_threshold or None,
        max_embed_tokens=int(max_embed_tokens) if max_embed_tokens else None,
        split_overlap_tokens=int(split_overlap_tokens) if split_overlap_tokens else None,
        azure_deployment=azure_deployment or None,
        azure_endpoint=azure_endpoint or None,
        azure_api_key=azure_api_key or None,
        medium_provider=medium_provider or None,
        truncate=truncate,
        dry_run=dry_run,
    )

    try:
        load_environment(loader_args.env_file)
        base_args = build_preprocess_args(loader_args)
        if loader_args.truncate:
            base_args.append("--truncate")
        step_args = select_step(loader_args)
        run_step(step_args=step_args, base_args=base_args)
    except Exception as exc:
        return f"Loader failed: {exc}"

    return "Loader completed successfully."


def _refresh_table_sources() -> str:
    try:
        with connect_from_env() as conn:
            mapping = inspect_table_sources(conn)
    except Exception as exc:
        return f"Unable to inspect database tables: {exc}"
    return format_table_sources(mapping)


with gr.Blocks(title="Knowledge Base Loader") as loader_app:
    gr.Markdown("# Document loader")

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", type="filepath")
        path_input = gr.Textbox(
            label="Or enter PDF path",
            placeholder="data/document.pdf",
            value="data/silmarillion.pdf",
        )

    source_text = gr.Textbox(label="Source name (optional)")

    table_radio = gr.Radio(
        ["mini", "small", "medium"],
        label="Target table",
        value="mini",
        info="Choose exactly one embedding table",
    )

    truncate_checkbox = gr.Checkbox(label="Truncate table before loading", value=False)
    dry_run_checkbox = gr.Checkbox(label="Dry run (skip DB writes)", value=False)

    with gr.Accordion("Chunking overrides", open=False):
        chunker_dropdown = gr.Dropdown(
            ["spacy", "semantic"],
            label="Chunker",
            value=DEFAULT_CHUNKER,
        )
        chunking_provider_dropdown = gr.Dropdown(
            ["local", "azure"],
            label="Chunking provider",
            value=DEFAULT_CHUNKING_PROVIDER,
        )
        chunking_local_model_text = gr.Textbox(
            label="Local model for semantic chunking",
            value=DEFAULT_CHUNKING_LOCAL_MODEL,
        )
        chunking_deployment_text = gr.Textbox(
            label="Azure deployment for chunking",
            value=DEFAULT_CHUNKING_DEPLOYMENT,
        )
        max_sentences_slider = gr.Slider(
            1,
            20,
            value=DEFAULT_MAX_SENTENCES,
            step=1,
            label="Max sentences per chunk",
        )
        spacy_model_text = gr.Textbox(label="spaCy model", value=DEFAULT_SPACY_MODEL)
        breakpoint_dropdown = gr.Dropdown(
            ["percentile", "standard_deviation", "interquartile"],
            label="Breakpoint threshold type",
            value=DEFAULT_BREAKPOINT,
        )
        max_embed_tokens_input = gr.Number(
            value=DEFAULT_MAX_EMBED_TOKENS,
            label="Max embed tokens (Azure)",
            precision=0,
        )
        split_overlap_input = gr.Number(
            value=DEFAULT_SPLIT_OVERLAP_TOKENS,
            label="Split overlap tokens (Azure)",
            precision=0,
        )

    with gr.Accordion("Medium embedding overrides", open=False):
        medium_provider_dropdown = gr.Dropdown(
            ["openai", "azure"],
            label="Medium embedding provider",
            value=DEFAULT_MEDIUM_PROVIDER,
            info="Choose Azure only if you have the endpoint and API key configured; otherwise OpenAI will be used.",
        )
        azure_deployment_text = gr.Textbox(label="Azure deployment", value=DEFAULT_AZURE_DEPLOYMENT)
        azure_endpoint_text = gr.Textbox(label="Azure endpoint", value=DEFAULT_AZURE_ENDPOINT)
        azure_api_key_text = gr.Textbox(label="Azure API key", value=DEFAULT_AZURE_API_KEY)

    run_button = gr.Button("Load document")
    status_output = gr.Textbox(label="Status")

    run_button.click(
        fn=run_loader,
        inputs=[
            pdf_upload,
            path_input,
            source_text,
            table_radio,
            chunker_dropdown,
            chunking_provider_dropdown,
            chunking_local_model_text,
            chunking_deployment_text,
            max_sentences_slider,
            spacy_model_text,
            breakpoint_dropdown,
            max_embed_tokens_input,
            split_overlap_input,
            azure_deployment_text,
            azure_endpoint_text,
            azure_api_key_text,
            medium_provider_dropdown,
            truncate_checkbox,
            dry_run_checkbox,
        ],
        outputs=status_output,
    )

    with gr.Accordion("Inspect knowledge base", open=False):
        refresh_button = gr.Button("Refresh tables & sources")
        table_sources_view = gr.Markdown(
            value="Click the button to list the existing tables and their distinct sources.",
            label="Tables and sources",
        )
        refresh_button.click(
            fn=_refresh_table_sources,
            inputs=[],
            outputs=table_sources_view,
        )


def launch_loader() -> None:
    loader_app.queue().launch(pwa=True)


if __name__ == "__main__":
    launch_loader()
