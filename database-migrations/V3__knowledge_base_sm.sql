CREATE TABLE knowledge_base_sm (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT,
    embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
