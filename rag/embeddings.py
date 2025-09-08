from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class Embedder:
    '''
    Embedder using SentenceTransformer model.
    Default model is 'intfloat/multilingual-e5-base' which supports multiple languages.'
    For e5 models, prepend "passage: " to passages and "query: " to queries.

    Args:
        model: str - The name of the SentenceTransformer model to use.
    '''
    def __init__(self, model: str='intfloat/multilingual-e5-base'):
        self.model = model
        self.transformer = SentenceTransformer(model)


    def embed_texts(self, chunks: list[Document]):
        texts = [doc.page_content for doc in chunks]
        if 'e5' in self.model.lower():
            texts = [f"passage: {doc.page_content}" for doc in chunks]
        return self.transformer.encode(texts)

    def embed_query(self, query: str):
        if 'e5' in self.model.lower():
            query = f"query: {query}"
        return self.transformer.encode([query])