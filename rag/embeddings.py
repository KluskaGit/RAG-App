from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import Iterator


from ollama import ChatResponse, chat


model_name = "intfloat/multilingual-e5-base"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

def save_to_vector_store(documents: list[Document], path: str) -> Chroma:

    # if 'e5' in model_name.lower():
    #     for doc in documents:
    #         doc.page_content = f'passage: {doc.page_content}'
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=path,
    )

    return vector_store


def load_chroma(path: str):
    return Chroma(persist_directory=path, embedding_function=embeddings)


def generate(query: str, prompt: str) -> ChatResponse:
    response = chat(
        model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF',
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}],
    )
    return response
