import chromadb
from rag.vectorstores.vectorstore import VectorStore

class ChromaCloud(VectorStore):
    
    """
        ChromaDB Cloud vector store client
        
        Args:
            database (str): Database name in ChromaDB Cloud
            api_key (str): API key for authentication
            tenant (str, optional): Tenant identifier.
                Optional. If not provided, it will be inferred from the API key if the key is scoped to a single tenant.
            collection_name (str): Name of the collection to use in ChromaDB
            embedding (EmbeddingFunction, optional): Embedding function to use. 
                Defaults to DefaultEmbeddingFunction().
    """

    def __init__(
            self,
            database: str,
            api_key: str,
            tenant: str | None = None,
            *args,
            **kwargs
        ):
        self.tenant = tenant
        self.database = database
        self.api_key = api_key
        super().__init__(*args, **kwargs)

    def _create_client(self):
        return chromadb.CloudClient(
            tenant=self.tenant,
            database=self.database,
            api_key=self.api_key
        )
        