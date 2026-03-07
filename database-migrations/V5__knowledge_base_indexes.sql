CREATE INDEX IF NOT EXISTS idx_kb_mini_embedding_hnsw
ON knowledge_base_mini
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_kb_sm_embedding_hnsw
ON knowledge_base_sm
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_kb_md_embedding_hnsw
ON knowledge_base_md
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
