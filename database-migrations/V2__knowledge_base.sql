
CREATE TABLE knowledge_base_mini (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
