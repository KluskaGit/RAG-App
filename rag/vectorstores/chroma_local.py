import chromadb
from rag.vectorstores.vectorstore import VectorStore

class ChromaLocal(VectorStore):

    """
        ChromaDB local vector store client

        Args:
            host (str): Host address of the ChromaDB server
            port (int): Port number of the ChromaDB server
            collection_name (str): Name of the collection to use in ChromaDB
            embedding (EmbeddingFunction, optional): Embedding function to use. 
                Defaults to DefaultEmbeddingFunction().
    """
    
    def __init__(
            self,
            host: str,
            port: int,
            *args,
            **kwargs
        ):
        self.host = host
        self.port = port
        super().__init__(*args, **kwargs)

    def _create_client(self):
        return chromadb.HttpClient(host=self.host, port=self.port)
        