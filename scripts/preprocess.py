from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Sequence

import pdfplumber
import psycopg2
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from langchain_openai import AzureOpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from utils.chunking import chunk_semantically_embeddings, chunk_semantically_spacy
from utils.embedding import SentenceTransformerWrapper


def extract_text_from_pdf(pdf_path: Path) -> str:
	text = ""
	with pdfplumber.open(pdf_path) as pdf:
		for page in pdf.pages:
			page_text = page.extract_text()
			if page_text:
				text += page_text + "\n"
	return text


def build_connection_string(args: argparse.Namespace) -> str:
	pg_user = args.pg_user or os.getenv("PGUSER")
	pg_password = args.pg_password or os.getenv("PGPASSWORD")
	pg_host = args.pg_host or os.getenv("PGHOST", "localhost")
	pg_port = str(args.pg_port or os.getenv("PGPORT", "5432"))
	pg_database = args.pg_database or os.getenv("PGDATABASE")

	missing = [
		name
		for name, value in {
			"PGUSER": pg_user,
			"PGPASSWORD": pg_password,
			"PGDATABASE": pg_database,
		}.items()
		if not value
	]
	if missing:
		raise ValueError(f"Missing PostgreSQL settings: {', '.join(missing)}")

	return (
		f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
	)


def embed_chunks_local(chunks: Sequence[str], model_name: str, device: str | None) -> List[List[float]]:
	wrapper = SentenceTransformerWrapper(model_name=model_name, device=device)
	return wrapper.embed_documents(list(chunks))


def embed_chunks_azure(chunks: Sequence[str], deployment_name: str, endpoint: str, api_key: str) -> List[List[float]]:
	base_url = endpoint.rstrip("/") + "/openai/v1/"
	client = OpenAI(base_url=base_url, api_key=api_key)

	embeddings: List[List[float]] = []
	for chunk in chunks:
		response = client.embeddings.create(input=chunk, model=deployment_name)
		embeddings.append(response.data[0].embedding)
	return embeddings


def embed_chunks_openai(chunks: Sequence[str], model_name: str, api_key: str) -> List[List[float]]:
	client = OpenAI(api_key=api_key)

	embeddings: List[List[float]] = []
	for chunk in chunks:
		response = client.embeddings.create(input=chunk, model=model_name)
		embeddings.append(response.data[0].embedding)
	return embeddings


def split_oversized_chunks(
	chunks: Sequence[str],
	max_tokens: int = 8000,
	overlap_tokens: int = 100,
) -> List[str]:
	encoder = tiktoken.get_encoding("cl100k_base")
	final_chunks: List[str] = []

	for chunk in chunks:
		tokens = encoder.encode(chunk)
		if len(tokens) <= max_tokens:
			final_chunks.append(chunk)
			continue

		step = max_tokens - overlap_tokens
		if step <= 0:
			raise ValueError("overlap_tokens must be smaller than max_tokens")

		for start in range(0, len(tokens), step):
			part = tokens[start : start + max_tokens]
			if not part:
				continue
			final_chunks.append(encoder.decode(part))

	return final_chunks


def get_embedding_chunker_for_provider(args: argparse.Namespace):
	if args.chunking_provider == "local":
		chunk_model = args.chunking_local_model or args.local_model
		return SentenceTransformerWrapper(model_name=chunk_model, device=args.device)

	deployment = args.chunking_deployment or args.deployment or os.getenv("DEPLOY_MEDIUM")
	endpoint = args.azure_endpoint or os.getenv("AZURE_ENDPOINT")
	api_key = args.azure_api_key or os.getenv("AZURE_API_KEY")
	api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
	if not deployment or not endpoint or not api_key:
		raise ValueError(
			"Azure semantic chunking requires deployment, endpoint, and api key "
			"(flags or env: DEPLOY_MEDIUM, AZURE_ENDPOINT, AZURE_API_KEY)."
		)

	return AzureOpenAIEmbeddings(
		model=deployment,
		azure_endpoint=endpoint,
		api_key=api_key,
		api_version=api_version,
	)


def insert_embeddings(
	conn_string: str,
	table_name: str,
	source_name: str,
	chunks: Sequence[str],
	embeddings: Sequence[Sequence[float]],
	truncate_first: bool,
) -> int:
	if len(chunks) != len(embeddings):
		raise ValueError("Chunk and embedding counts do not match")

	rows = [(source_name, chunk, embedding) for chunk, embedding in zip(chunks, embeddings)]
	query = f"INSERT INTO {table_name} (source, content, embedding) VALUES %s"

	with psycopg2.connect(conn_string) as conn:
		with conn.cursor() as cur:
			register_vector(cur)
			if truncate_first:
				cur.execute(f"TRUNCATE TABLE {table_name}")
			execute_values(cur, query, rows)
			conn.commit()
	return len(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Read PDF, chunk text, create embeddings, and populate pgvector table."
	)
	parser.add_argument("--pdf-path", default="data/euaiact.pdf", help="Path to PDF file")
	parser.add_argument("--source", default=None, help="Source name stored in DB (default: PDF filename)")
	parser.add_argument("--table", required=True, help="Target table (e.g., knowledge_base_sm)")
	parser.add_argument("--chunker", choices=["spacy", "semantic"], default="spacy", help="Chunking strategy")
	parser.add_argument(
		"--chunking-provider",
		choices=["local", "azure"],
		default="local",
		help="Embedding provider used only for semantic chunk boundary detection",
	)
	parser.add_argument(
		"--chunking-local-model",
		default=None,
		help="Local model for semantic chunk boundary detection (defaults to --local-model)",
	)
	parser.add_argument(
		"--chunking-deployment",
		default=None,
		help="Azure deployment for semantic chunk boundary detection (defaults to --deployment)",
	)
	parser.add_argument("--max-sentences", type=int, default=5, help="Sentences per chunk for spaCy chunker")
	parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name")
	parser.add_argument(
		"--breakpoint-threshold-type",
		choices=["percentile", "standard_deviation", "interquartile"],
		default="percentile",
		help="Threshold strategy for embedding-based semantic chunking",
	)
	parser.add_argument("--provider", choices=["local", "azure", "openai"], default="local", help="Embedding provider")
	parser.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
	parser.add_argument("--device", default=None, help="Device for local model (cpu/cuda)")
	parser.add_argument("--deployment", default=None, help="Embedding deployment or model name")
	parser.add_argument("--azure-endpoint", default=None, help="Azure endpoint override (otherwise uses AZURE_ENDPOINT)")
	parser.add_argument("--azure-api-key", default=None, help="Azure API key override (otherwise uses AZURE_API_KEY)")
	parser.add_argument("--openai-api-key", default=None, help="OpenAI API key override (otherwise uses OPENAI_API_KEY)")
	parser.add_argument("--truncate", action="store_true", help="Truncate target table before inserting")
	parser.add_argument("--dry-run", action="store_true", help="Only print counts; do not write to DB")
	parser.add_argument("--max-embed-tokens", type=int, default=8000, help="Max tokens per chunk for Azure embedding requests")
	parser.add_argument("--split-overlap-tokens", type=int, default=50, help="Token overlap when splitting oversized chunks for Azure")

	parser.add_argument("--pg-user", default=None)
	parser.add_argument("--pg-password", default=None)
	parser.add_argument("--pg-host", default=None)
	parser.add_argument("--pg-port", default=None)
	parser.add_argument("--pg-database", default=None)
	return parser.parse_args()


def main() -> None:
	load_dotenv("project.env")
	load_dotenv(".env")

	args = parse_args()
	pdf_path = Path(args.pdf_path)
	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF not found: {pdf_path}")

	source_name = args.source or pdf_path.name
	pdf_text = extract_text_from_pdf(pdf_path)

	if args.chunker == "spacy":
		chunks = chunk_semantically_spacy(
			pdf_text,
			max_sentences=args.max_sentences,
			spacy_model=args.spacy_model,
		)
	else:
		chunker_embeddings = get_embedding_chunker_for_provider(args)
		chunks = chunk_semantically_embeddings(
			pdf_text,
			embeddings=chunker_embeddings,
			breakpoint_threshold_type=args.breakpoint_threshold_type,
		)

	if args.provider == "local":
		embeddings = embed_chunks_local(chunks=chunks, model_name=args.local_model, device=args.device)
	elif args.provider == "azure":
		chunks = split_oversized_chunks(
			chunks=chunks,
			max_tokens=args.max_embed_tokens,
			overlap_tokens=args.split_overlap_tokens,
		)
		deployment = args.deployment or os.getenv("DEPLOY_MEDIUM")
		endpoint = args.azure_endpoint or os.getenv("AZURE_ENDPOINT")
		api_key = args.azure_api_key or os.getenv("AZURE_API_KEY")
		if not deployment or not endpoint or not api_key:
			raise ValueError(
				"Azure embedding requires deployment, endpoint, and api key "
				"(flags or env: DEPLOY_MEDIUM, AZURE_ENDPOINT, AZURE_API_KEY)."
			)
		embeddings = embed_chunks_azure(
			chunks=chunks,
			deployment_name=deployment,
			endpoint=endpoint,
			api_key=api_key,
		)
	elif args.provider == "openai":
		chunks = split_oversized_chunks(
			chunks=chunks,
			max_tokens=args.max_embed_tokens,
			overlap_tokens=args.split_overlap_tokens,
		)
		model_name = args.deployment or os.getenv("DEPLOY_MEDIUM")
		api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
		if not model_name or not api_key:
			raise ValueError(
				"OpenAI embedding requires deployment/model name and OPENAI_API_KEY. "
				"Set DEPLOY_MEDIUM and OPENAI_API_KEY in env or pass them via flags."
			)
		embeddings = embed_chunks_openai(
			chunks=chunks,
			model_name=model_name,
			api_key=api_key,
		)

	dim = len(embeddings[0]) if embeddings else 0
	print(f"PDF: {pdf_path}")
	print(f"Chunks: {len(chunks)}")
	print(f"Chunker: {args.chunker}")
	print(f"Embedding dim: {dim}")
	print(f"Target table: {args.table}")

	if args.dry_run:
		print("Dry run enabled, skipping DB insert.")
		return

	conn_string = build_connection_string(args)
	inserted = insert_embeddings(
		conn_string=conn_string,
		table_name=args.table,
		source_name=source_name,
		chunks=chunks,
		embeddings=embeddings,
		truncate_first=args.truncate,
	)
	print(f"Inserted rows: {inserted}")


if __name__ == "__main__":
	main()
