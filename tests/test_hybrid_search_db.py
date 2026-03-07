from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    load_dotenv(ROOT / "project.env", override=False)
    load_dotenv(ROOT / ".env", override=True)


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _build_dsn() -> str:
    host = _env_or_default("PGHOST", "localhost")
    port = _env_or_default("PGPORT", "5431")
    database = _env_or_default("PGDATABASE", "postgres")
    user = _env_or_default("PGUSER", "postgres")
    password = _env_or_default("PGPASSWORD", "password")
    sslmode = _env_or_default("PGSSLMODE", "disable")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


class TestHybridSearchDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_env()
        cls.conn = psycopg2.connect(_build_dsn())
        cls.conn.autocommit = True

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "conn", None):
            cls.conn.close()

    def _fetchone_value(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return None if row is None else row[0]

    def test_expected_hybrid_objects_exist(self) -> None:
        expected_columns = [
            ("knowledge_base_mini", "content_tsv"),
            ("knowledge_base_sm", "content_tsv"),
            ("knowledge_base_md", "content_tsv"),
        ]
        for table_name, column_name in expected_columns:
            exists = self._fetchone_value(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = %s
                      AND column_name = %s
                )
                """,
                (table_name, column_name),
            )
            self.assertTrue(exists, f"Missing column {table_name}.{column_name}")

        expected_functions = [
            "kb_rrf_score",
            "search_knowledge_base_mini_hybrid",
            "search_knowledge_base_sm_hybrid",
            "search_knowledge_base_md_hybrid",
        ]
        for function_name in expected_functions:
            exists = self._fetchone_value(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_proc p
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public'
                      AND p.proname = %s
                )
                """,
                (function_name,),
            )
            self.assertTrue(exists, f"Missing function {function_name}")

    def _assert_hybrid_query(self, table_name: str, function_name: str, embedding_dim: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT content, embedding::text
                FROM {table_name}
                WHERE embedding IS NOT NULL AND content IS NOT NULL
                LIMIT 1
                """
            )
            sample = cur.fetchone()
            if sample is None:
                self.skipTest(
                    f"Table {table_name} has no populated rows yet. "
                    "Populate DB before running integration tests."
                )

            sample_content, sample_embedding_text = sample
            query_text = " ".join(str(sample_content).split()[:8]).strip() or "rag"

            cur.execute(
                f"""
                SELECT id, source, content, vector_similarity, text_score, rrf_score
                FROM {function_name}(%s, %s::vector({embedding_dim}), 10, 100, 60)
                """,
                (query_text, sample_embedding_text),
            )
            rows = cur.fetchall()

        self.assertGreater(len(rows), 0, f"Hybrid function {function_name} returned no rows")

        previous_rrf = None
        for row in rows:
            self.assertIsNotNone(row[0], "id should not be NULL")
            self.assertIsNotNone(row[1], "source should not be NULL")
            self.assertIsNotNone(row[2], "content should not be NULL")
            self.assertIsNotNone(row[5], "rrf_score should not be NULL")
            if previous_rrf is not None:
                self.assertGreaterEqual(previous_rrf, row[5], "Results are not ordered by rrf_score DESC")
            previous_rrf = row[5]

    def test_hybrid_query_mini(self) -> None:
        self._assert_hybrid_query(
            table_name="knowledge_base_mini",
            function_name="search_knowledge_base_mini_hybrid",
            embedding_dim=384,
        )

    def test_hybrid_query_sm(self) -> None:
        self._assert_hybrid_query(
            table_name="knowledge_base_sm",
            function_name="search_knowledge_base_sm_hybrid",
            embedding_dim=1024,
        )

    def test_hybrid_query_md(self) -> None:
        self._assert_hybrid_query(
            table_name="knowledge_base_md",
            function_name="search_knowledge_base_md_hybrid",
            embedding_dim=1536,
        )


if __name__ == "__main__":
    print("Running TestHybridSearchDatabase...\n")
    print("Note: These tests require a populated database with the expected schema and functions. ")
    print("Make sure to run the loader app to populate the database before running these tests.")
    print("If you encounter failures, check the database connection settings and ensure the loader has been run successfully.\n\n")
    unittest.main(verbosity=2)
