CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE OR REPLACE FUNCTION kb_update_content_tsv() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

ALTER TABLE knowledge_base_mini ADD COLUMN IF NOT EXISTS content_tsv tsvector;
ALTER TABLE knowledge_base_sm ADD COLUMN IF NOT EXISTS content_tsv tsvector;
ALTER TABLE knowledge_base_md ADD COLUMN IF NOT EXISTS content_tsv tsvector;

UPDATE knowledge_base_mini
SET content_tsv = to_tsvector('english', COALESCE(content, ''))
WHERE content_tsv IS NULL;

UPDATE knowledge_base_sm
SET content_tsv = to_tsvector('english', COALESCE(content, ''))
WHERE content_tsv IS NULL;

UPDATE knowledge_base_md
SET content_tsv = to_tsvector('english', COALESCE(content, ''))
WHERE content_tsv IS NULL;

DROP TRIGGER IF EXISTS trg_kb_mini_content_tsv ON knowledge_base_mini;
CREATE TRIGGER trg_kb_mini_content_tsv
BEFORE INSERT OR UPDATE OF content ON knowledge_base_mini
FOR EACH ROW
EXECUTE FUNCTION kb_update_content_tsv();

DROP TRIGGER IF EXISTS trg_kb_sm_content_tsv ON knowledge_base_sm;
CREATE TRIGGER trg_kb_sm_content_tsv
BEFORE INSERT OR UPDATE OF content ON knowledge_base_sm
FOR EACH ROW
EXECUTE FUNCTION kb_update_content_tsv();

DROP TRIGGER IF EXISTS trg_kb_md_content_tsv ON knowledge_base_md;
CREATE TRIGGER trg_kb_md_content_tsv
BEFORE INSERT OR UPDATE OF content ON knowledge_base_md
FOR EACH ROW
EXECUTE FUNCTION kb_update_content_tsv();

CREATE INDEX IF NOT EXISTS idx_kb_mini_content_tsv_gin
ON knowledge_base_mini
USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS idx_kb_sm_content_tsv_gin
ON knowledge_base_sm
USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS idx_kb_md_content_tsv_gin
ON knowledge_base_md
USING gin (content_tsv);
