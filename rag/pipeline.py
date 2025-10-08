# from rag.loaders import load_file
# from rag.splitters import chunk_documents
# #from langchain_community.llms import HuggingFaceHub
# from rag.embeddings import save_to_vector_store, load_chroma, generate

# def pipe(query: str):
#     file = load_file('data/Zalacznik-Nr-5.pdf')
#     all_splits = chunk_documents(file)

#     vectore_store = load_chroma('vectordb')#save_to_vector_store(all_splits, path="vectordb")
#     results = vectore_store.similarity_search(query, k=3)

#     prompt = f'''
#     You are a helpful chatbot.
#     Use only the following pieces of context to answer the question. Don't make up any new information:
#     {'\n'.join([doc.page_content for doc in results])}
#     '''
#     print(prompt)
#     return generate(query, prompt).message.content

from rag.loaders import FileLoader
from rag.splitters import TokenTextSplitter
from rag.vectorstore import VectorStore

loader = FileLoader()
doc = loader.load_file('data/Zalacznik-Nr-5.pdf')
splitter = TokenTextSplitter(chunk_size=500, overlap=50)
chunks = splitter.split_documents(doc)
vectoreStore = VectorStore('vectordb')
vectoreStore.save_documents(chunks)