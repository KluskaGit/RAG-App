import chromadb
import uuid
from chromadb.api.types import Embeddings, Metadata
from rag.schemas.document import Document

class VectorStore:

    def __init__(
            self,
            persist_directory: str
        ):

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="documents")

    def save_texts(
            self,
            texts: list[str],
            #embeddings: Embeddings,
            metadatas: list[Metadata] | None = None,
            
        ):
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in texts],
            documents=texts,
            metadatas=metadatas,
            #embeddings=embeddings
        )

    def save_documents(self, documents: list[Document]):
        texts: list[str] = []
        metadatas: list[Metadata] = []
        for doc in documents:
            texts.append(doc.text)
            metadatas.append(doc.metadata if doc.metadata else {})

        self.save_texts(texts, metadatas)