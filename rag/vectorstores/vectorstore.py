import uuid
from abc import ABC, abstractmethod
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction, Metadata, QueryResult
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rag.schemas.document import Document

class VectorStore(ABC):
    """
        Base class for ChromaDB vector store clients

        Args:
            collection_name (str): Name of the collection to use in ChromaDB
            embedding (EmbeddingFunction, optional): Embedding function to use. 
                Defaults to DefaultEmbeddingFunction().
    """

    def __init__(
            self,
            collection_name: str,
            embedding: EmbeddingFunction = DefaultEmbeddingFunction(),
        ):

        self.client = self._create_client()
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=embedding)

    @abstractmethod
    def _create_client(self) -> ClientAPI:
        """
            Returns a ChromaDB client
        """
        pass

    def save_texts(
            self,
            texts: list[str],
            metadatas: list[Metadata] | None = None,
            
        ):
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in texts],
            documents=texts,
            metadatas=metadatas,
        )

    def save_documents(self, documents: list[Document]):
        texts: list[str] = []
        metadatas: list[Metadata] = []
        for doc in documents:
            texts.append(doc.text)
            metadatas.append(doc.metadata if doc.metadata else {})

        self.save_texts(texts, metadatas)

    def similarity_search(
            self,
            query: str,
            k: int = 3
        ) -> QueryResult:

        return self.collection.query(
            query_texts=query,
            n_results=k
        )