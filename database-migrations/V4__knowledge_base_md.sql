CREATE TABLE knowledge_base_md (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
