from loaders import load_file
from splitters import chunk_documents
#from langchain_community.llms import HuggingFaceHub
from embeddings import save_to_vector_store, load_chroma, generate


file = load_file('../data/Zalacznik-Nr-5.pdf')
all_splits = chunk_documents(file)

vectore_store = load_chroma('../vectordb')#save_to_vector_store(all_splits)
query = 'Jakie kryteria trzeba spełnić aby otrzymać stypendium rektora'
results = vectore_store.similarity_search(query, k=3)

prompt = f'''
You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([doc.page_content for doc in results])}
'''

print(generate(query, prompt))





