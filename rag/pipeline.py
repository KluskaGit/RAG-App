import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.loaders import FileLoader
from rag.splitters import TokenTextSplitter
from rag.vectorstore import VectorStore

class Pipeline:
    def __init__(
            self,
            embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
        ):
        self.loader = FileLoader()
        self.splitter = TokenTextSplitter(chunk_size=500, overlap=150)
        self.huggingface_ef = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.vectoreStore = VectorStore('documents', 'vectordb', embedding=self.huggingface_ef)

    def save_data(self, folder_path: str) -> None:
        file_names: list[str] = os.listdir(folder_path)

        for file in file_names:
            file_path = os.path.join(folder_path, file)
            documents = self.loader.load_file(file_path)
            chunks = self.splitter.split_documents(documents)
            self.vectoreStore.save_documents(chunks)

    def generate_response(self, query: str) -> tuple[str, list]:
        result = self.vectoreStore.similarity_search(query=query)
        if context:=result.get('documents', None):
            context = '\n'.join(context[0])
        else:
            context = "Brak wyników"

        if metadata:= result.get('metadatas', None):
            metadata = metadata[0]
        else:
            metadata = []
        return context, metadata
        
