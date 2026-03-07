CREATE OR REPLACE FUNCTION kb_rrf_score(rank_position bigint, rrf_k integer DEFAULT 60)
RETURNS double precision
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT 1.0 / (rrf_k + rank_position);
$$;

CREATE OR REPLACE FUNCTION search_knowledge_base_mini_hybrid(
    query_text text,
    query_embedding vector(384),
    top_k integer DEFAULT 20,
    candidate_k integer DEFAULT 100,
    rrf_k integer DEFAULT 60
)
RETURNS TABLE (
    id integer,
    source text,
    content text,
    vector_similarity double precision,
    text_score double precision,
    rrf_score double precision
)
LANGUAGE sql
STABLE
AS $$
WITH vector_hits AS (
    SELECT
        kb.id,
        kb.source,
        kb.content,
        1.0 - (kb.embedding <=> query_embedding) AS vector_similarity,
        ROW_NUMBER() OVER (ORDER BY kb.embedding <=> query_embedding, kb.id) AS vector_rank
    FROM knowledge_base_mini kb
    ORDER BY kb.embedding <=> query_embedding, kb.id
    LIMIT candidate_k
),
text_hits AS (
    SELECT
        kb.id,
        kb.source,
        kb.content,
        ts_rank_cd(kb.content_tsv, websearch_to_tsquery('english', query_text)) AS text_score,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(kb.content_tsv, websearch_to_tsquery('english', query_text)) DESC, kb.id
        ) AS text_rank
    FROM knowledge_base_mini kb
    WHERE kb.content_tsv @@ websearch_to_tsquery('english', query_text)
    ORDER BY text_score DESC, kb.id
    LIMIT candidate_k
),
fused AS (
    SELECT
        COALESCE(v.id, t.id) AS id,
        COALESCE(v.source, t.source) AS source,
        COALESCE(v.content, t.content) AS content,
        v.vector_similarity,
        t.text_score,
        COALESCE(kb_rrf_score(v.vector_rank, rrf_k), 0.0) + COALESCE(kb_rrf_score(t.text_rank, rrf_k), 0.0) AS rrf_score
    FROM vector_hits v
    FULL OUTER JOIN text_hits t USING (id)
)
SELECT id, source, content, vector_similarity, text_score, rrf_score
FROM fused
ORDER BY rrf_score DESC, id
LIMIT top_k;
$$;

CREATE OR REPLACE FUNCTION search_knowledge_base_sm_hybrid(
    query_text text,
    query_embedding vector(1024),
    top_k integer DEFAULT 20,
    candidate_k integer DEFAULT 100,
    rrf_k integer DEFAULT 60
)
RETURNS TABLE (
    id integer,
    source text,
    content text,
    vector_similarity double precision,
    text_score double precision,
    rrf_score double precision
)
LANGUAGE sql
STABLE
AS $$
WITH vector_hits AS (
    SELECT
        kb.id,
        kb.source,
        kb.content,
        1.0 - (kb.embedding <=> query_embedding) AS vector_similarity,
        ROW_NUMBER() OVER (ORDER BY kb.embedding <=> query_embedding, kb.id) AS vector_rank
    FROM knowledge_base_sm kb
    ORDER BY kb.embedding <=> query_embedding, kb.id
    LIMIT candidate_k
),
text_hits AS (
    SELECT
        kb.id,
        kb.source,
        kb.content,
        ts_rank_cd(kb.content_tsv, websearch_to_tsquery('english', query_text)) AS text_score,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(kb.content_tsv, websearch_to_tsquery('english', query_text)) DESC, kb.id
        ) AS text_rank
    FROM knowledge_base_sm kb
    WHERE kb.content_tsv @@ websearch_to_tsquery('english', query_text)
    ORDER BY text_score DESC, kb.id
    LIMIT candidate_k
),
fused AS (
    SELECT
        COALESCE(v.id, t.id) AS id,
        COALESCE(v.source, t.source) AS source,
        COALESCE(v.content, t.content) AS content,
        v.vector_similarity,
        t.text_score,
        COALESCE(kb_rrf_score(v.vector_rank, rrf_k), 0.0) + COALESCE(kb_rrf_score(t.text_rank, rrf_k), 0.0) AS rrf_score
    FROM vector_hits v
    FULL OUTER JOIN text_hits t USING (id)
)
SELECT id, source, content, vector_similarity, text_score, rrf_score
FROM fused
ORDER BY rrf_score DESC, id
LIMIT top_k;
$$;

CREATE OR REPLACE FUNCTION search_knowledge_base_md_hybrid(
    query_text text,
    query_embedding vector(1536),
    top_k integer DEFAULT 20,
    candidate_k integer DEFAULT 100,
    rrf_k integer DEFAULT 60
)
RETURNS TABLE (
    id integer,
    source text,
    content text,
    vector_similarity double precision,
    text_score double precision,
    rrf_score double precision
)
LANGUAGE sql
STABLE
AS $$
WITH vector_hits AS (
    SELECT
        kb.id,
        kb.source,
        kb.content,
        1.0 - (kb.embedding <=> query_embedding) AS vector_similarity,
        ROW_NUMBER() OVER (ORDER BY kb.embedding <=> query_embedding, kb.id) AS vector_rank
    FROM knowledge_base_md kb
    ORDER BY kb.embedding <=> query_embedding, kb.id
    LIMIT candidate_k
),
text_hits AS (
    SELECT
        kb.id,
        kb.source,
        kb.content,
        ts_rank_cd(kb.content_tsv, websearch_to_tsquery('english', query_text)) AS text_score,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(kb.content_tsv, websearch_to_tsquery('english', query_text)) DESC, kb.id
        ) AS text_rank
    FROM knowledge_base_md kb
    WHERE kb.content_tsv @@ websearch_to_tsquery('english', query_text)
    ORDER BY text_score DESC, kb.id
    LIMIT candidate_k
),
fused AS (
    SELECT
        COALESCE(v.id, t.id) AS id,
        COALESCE(v.source, t.source) AS source,
        COALESCE(v.content, t.content) AS content,
        v.vector_similarity,
        t.text_score,
        COALESCE(kb_rrf_score(v.vector_rank, rrf_k), 0.0) + COALESCE(kb_rrf_score(t.text_rank, rrf_k), 0.0) AS rrf_score
    FROM vector_hits v
    FULL OUTER JOIN text_hits t USING (id)
)
SELECT id, source, content, vector_similarity, text_score, rrf_score
FROM fused
ORDER BY rrf_score DESC, id
LIMIT top_k;
$$;
