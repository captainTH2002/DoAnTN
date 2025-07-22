-- Tạo extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tạo bảng documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embedding dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo index cho vector search
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops); 