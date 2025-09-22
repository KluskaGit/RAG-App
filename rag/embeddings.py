import huggingface_hub
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import ollama


model_name = "intfloat/multilingual-e5-base"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

def save_to_vector_store(documents: list[Document]):

    # if 'e5' in model_name.lower():
    #     for doc in documents:
    #         doc.page_content = f'passage: {doc.page_content}'
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="../vectordb",
    )

    return vector_store


def load_chroma(path: str):
    return Chroma(persist_directory=path, embedding_function=embeddings)


def generate(query: str, prompt: str):
    response = ollama.chat(
        model='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF',
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}]
    )
    return response
