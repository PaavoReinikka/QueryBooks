from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Literal, Mapping, Sequence, Optional

# NOTE: psycopg2 is used for compatibility with other projects
# while psycopg (v3) should be preferred for new development when possible. 
import psycopg2
from psycopg2 import Error as PsycopgError, extensions, sql, errorcodes

from utils.loader_utils import env_or_default

Profile = Literal["mini", "sm", "md"]


@dataclass(frozen=True)
class RetrievalRow:
    id: int
    source: str
    content: str
    vector_similarity: float | None = None
    text_score: float | None = None
    rrf_score: float | None = None


_PROFILE_TABLE_DIM: Mapping[Profile, tuple[str, int]] = {
    "mini": ("knowledge_base_mini", 384),
    "sm": ("knowledge_base_sm", 1024),
    "md": ("knowledge_base_md", 1536),
}


def _profile_meta(profile: Profile) -> tuple[str, int]:
    return _PROFILE_TABLE_DIM[profile]


_DEFAULT_TABLE_SCHEMA = "public"
_TABLE_PREFIX = "knowledge_base_"


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _rows_to_objects(rows: Sequence[tuple[Any, ...]]) -> list[RetrievalRow]:
    return [
        RetrievalRow(
            id=row[0],
            source=row[1],
            content=row[2],
            vector_similarity=row[3] if len(row) > 3 else None,
            text_score=row[4] if len(row) > 4 else None,
            rrf_score=row[5] if len(row) > 5 else None,
        )
        for row in rows
    ]


def retrieval_rows_to_dicts(rows: Sequence[RetrievalRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def semantic_retrieve(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
    query_embedding: Sequence[float],
    top_k: int = 8,
    source_filter: Optional[str] = None,
) -> list[RetrievalRow]:
    table_name, vector_dim = _profile_meta(profile)
    vector_list = _as_float_list(query_embedding)

    sql = f"""
        SELECT
            id,
            source,
            content,
            1.0 - (embedding <=> %s::vector({vector_dim})) AS vector_similarity,
            NULL::double precision AS text_score,
            NULL::double precision AS rrf_score
        FROM {table_name}
        WHERE (%s IS NULL OR source = %s)
        ORDER BY embedding <=> %s::vector({vector_dim}), id
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute("SAVEPOINT semantic_vector_bind")
        try:
            cur.execute(sql, (source_filter, source_filter, vector_list, vector_list, top_k))
            rows = cur.fetchall()
            cur.execute("RELEASE SAVEPOINT semantic_vector_bind")
        except PsycopgError:
            vector_literal = _vector_literal(query_embedding)
            cur.execute("ROLLBACK TO SAVEPOINT semantic_vector_bind")
            cur.execute(sql, (source_filter, source_filter, vector_literal, vector_literal, top_k))
            rows = cur.fetchall()
            cur.execute("RELEASE SAVEPOINT semantic_vector_bind")
    return _rows_to_objects(rows)


def lexical_retrieve(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
    query_text: str,
    top_k: int = 8,
    ts_config: str = "english",
    source_filter: Optional[str] = None,
) -> list[RetrievalRow]:
    table_name, _ = _profile_meta(profile)

    sql = f"""
        SELECT
            id,
            source,
            content,
            NULL::double precision AS vector_similarity,
            ts_rank_cd(content_tsv, websearch_to_tsquery(%s, %s)) AS text_score,
            NULL::double precision AS rrf_score
        FROM {table_name}
        WHERE (%s IS NULL OR source = %s)
        AND content_tsv @@ websearch_to_tsquery(%s, %s)
        ORDER BY text_score DESC, id
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                source_filter,
                source_filter,
                ts_config,
                query_text,
                ts_config,
                query_text,
                top_k,
            ),
        )
        rows = cur.fetchall()
    return _rows_to_objects(rows)


def hybrid_retrieve(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
    query_text: str,
    query_embedding: Sequence[float],
    top_k: int = 8,
    candidate_k: int = 100,
    rrf_k: int = 60,
    source_filter: Optional[str] = None,
) -> list[RetrievalRow]:
    _, vector_dim = _profile_meta(profile)
    vector_list = _as_float_list(query_embedding)

    function_name = f"search_knowledge_base_{profile}_hybrid"
    sql = f"""
        SELECT id, source, content, vector_similarity, text_score, rrf_score
        FROM {function_name}(%s, %s::vector({vector_dim}), %s, %s, %s) AS t
        WHERE (%s IS NULL OR t.source = %s)
    """

    with conn.cursor() as cur:
        cur.execute("SAVEPOINT hybrid_vector_bind")
        try:
            cur.execute(
                sql,
                (
                    query_text,
                    vector_list,
                    top_k,
                    candidate_k,
                    rrf_k,
                    source_filter,
                    source_filter,
                ),
            )
            rows = cur.fetchall()
            cur.execute("RELEASE SAVEPOINT hybrid_vector_bind")
        except PsycopgError:
            vector_literal = _vector_literal(query_embedding)
            cur.execute("ROLLBACK TO SAVEPOINT hybrid_vector_bind")
            cur.execute(
                sql,
                (
                    query_text,
                    vector_literal,
                    top_k,
                    candidate_k,
                    rrf_k,
                    source_filter,
                    source_filter,
                ),
            )
            rows = cur.fetchall()
            cur.execute("RELEASE SAVEPOINT hybrid_vector_bind")
    return _rows_to_objects(rows)


def list_sources(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
) -> list[str]:
    table_name, _ = _profile_meta(profile)
    sql = f"SELECT DISTINCT source FROM {table_name} WHERE source IS NOT NULL ORDER BY source"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [row[0] for row in rows]


def open_connection(dsn: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(dsn)


def build_postgres_dsn() -> str:
    """Build a libpq connection string from the configured PostgreSQL env vars."""

    parts = [
        f"dbname={env_or_default('PGDATABASE', 'postgres')}",
        f"user={env_or_default('PGUSER', 'postgres')}",
        f"password={env_or_default('PGPASSWORD', 'password')}",
        f"host={env_or_default('PGHOST', 'database')}",
        f"port={env_or_default('PGPORT', '5432')}",
    ]
    sslmode = os.getenv('PGSSLMODE')
    if sslmode:
        parts.append(f"sslmode={sslmode}")
    return " ".join(parts)


def connect_from_env() -> extensions.connection:
    """Open a psycopg2 connection using the current environment."""

    return open_connection(build_postgres_dsn())


def list_tables(
    conn: extensions.connection,
    *,
    schema: str = _DEFAULT_TABLE_SCHEMA,
    prefix: str = _TABLE_PREFIX,
) -> list[str]:
    """Return every table in *schema* whose name begins with *prefix*."""

    with conn.cursor() as cur:
        cur.execute(
            (
                "SELECT table_name"
                " FROM information_schema.tables"
                " WHERE table_schema = %s"
                " AND table_type = 'BASE TABLE'"
                " AND table_name LIKE %s"
                " ORDER BY table_name"
            ),
            (schema, f"{prefix}%"),
        )
        rows = cur.fetchall()
    return [row[0] for row in rows]


def inspect_table_sources(
    conn: extensions.connection,
    *,
    table_names: Sequence[str] | None = None,
    schema: str = _DEFAULT_TABLE_SCHEMA,
    prefix: str = _TABLE_PREFIX,
) -> dict[str, list[str]]:
    """Return the distinct "source" values for each table (defaults to the knowledge-base tables)."""

    names = table_names or list_tables(conn, schema=schema, prefix=prefix)
    mapping: dict[str, list[str]] = {}
    with conn.cursor() as cur:
        for table in names:
            stmt = sql.SQL(
                "SELECT DISTINCT source"
                " FROM {}"
                " WHERE source IS NOT NULL"
                " ORDER BY source"
            ).format(sql.Identifier(table))
            try:
                cur.execute(stmt)
            except PsycopgError as exc:
                if exc.pgcode == errorcodes.UNDEFINED_COLUMN:
                    mapping[table] = []
                    continue
                raise
            mapping[table] = [row[0] for row in cur.fetchall()]
    return mapping


def format_table_sources(
    mapping: Mapping[str, Sequence[str]], *, markdown: bool = True
) -> str:
    """Return a human-friendly string describing each table's sources."""

    if not mapping:
        return "No tables found."

    lines: list[str] = []
    for table, sources in mapping.items():
        label = f"**{table}**" if markdown else table
        count = len(sources)
        contents = ", ".join(sources) if sources else "(no sources recorded yet)"
        lines.append(f"{label} ({count} sources): {contents}")
    return "\n\n".join(lines)
