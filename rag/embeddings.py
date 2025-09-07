from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class Embedder:
    def __init__(self, model: str='intfloat/multilingual-e5-base'):
        self.model = SentenceTransformer(model)

    def embed_texts(self, chunks: list[Document]):
        texts = [f"passage: {doc.page_content}" for doc in chunks]
        return self.model.encode(texts)

    def embed_query(self, query: str):
        return self.model.encode([f"query: {query}"])