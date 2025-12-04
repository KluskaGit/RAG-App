import os
import yaml
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

from rag.loaders import FileLoader
from rag.splitters import TokenTextSplitter
from rag.vectorstores.chroma_local import ChromaLocal
from rag.vectorstores.chroma_cloud import ChromaCloud
from rag.embeddings.embedding import get_embedding_function

class Pipeline:
    def __init__(self):

        load_dotenv()
        with open(Path('appconfig.yaml'), 'r') as cfg:
            self.config = yaml.safe_load(cfg)

        self.loader = FileLoader()
        self.splitter = TokenTextSplitter(
            chunk_size=self.config['text-splitter']['chunk_size'],
            overlap=self.config['text-splitter']['overlap'])

        embedding = get_embedding_function(**self.config['retriever'])

        if self.config['vectorstore']['chroma_client'] == "local":
            self.vectorStore = ChromaLocal(
                host = self.config['vectorstore']['host'],
                port = self.config['vectorstore']['port'],
                collection_name = self.config['vectorstore']['collection_name'],
                embedding=embedding
            )
        elif self.config['vectorstore']['chroma_client'] == "cloud":

            if tenant:=self.config['vectorstore'].get('tenant', None):
                tenant = os.environ[tenant]

            self.vectorStore = ChromaCloud(
                database = self.config['vectorstore']['database'],
                api_key=os.environ[self.config['vectorstore']['api_key']],
                tenant = tenant,
                collection_name = self.config['vectorstore']['collection_name'],
                embedding=embedding
            )
        else:
            raise ValueError("Unsupported chroma client type in configuration.")

        self.save_data(folder_path='data')
        
    def save_data(self, folder_path: str) -> None:
        file_names: list[str] = os.listdir(folder_path)

        for file in file_names:
            file_path = os.path.join(folder_path, file)
            documents = self.loader.load_file(file_path)
            chunks = self.splitter.split_documents(documents)
            self.vectorStore.save_documents(chunks)

    def generate_context(self, query: str) -> tuple[str, list]:
        result = self.vectorStore.similarity_search(query=query, k=self.config['text-splitter']['top_k'])
        if context:=result.get('documents', None):
            context = '\n'.join(context[0])
        else:
            context = "Brak wyników"

        if metadata:= result.get('metadatas', None):
            metadata = metadata[0]
        else:
            metadata = []
        return context, metadata
    
    def generate_response(self, prompt: str) -> tuple[str | None, list]:
        context, metadata = self.generate_context(query=prompt)
        
        system_prompt = f"""
            You are a helpful chatbot.
            Use only the following pieces of context to answer the question. Don't make up any new information:
            {context}
        """
        client = OpenAI(
            base_url=self.config['LLM']['base_url'],
            api_key=os.environ[self.config['LLM']['api_key']],
        )
        
        response = client.chat.completions.create(
            model=self.config['LLM']['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Answer the question based on the context: {prompt}"},
            ]
        )
        message = response.choices[0].message.content

        return message, metadata