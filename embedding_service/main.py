from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # download lần đầu

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
def embed_text(req: TextRequest):
    embedding = model.encode(req.text).tolist()
    return {"embedding": embedding}
# uvicorn main:app --reload --host 0.0.0.0 --port 8001 