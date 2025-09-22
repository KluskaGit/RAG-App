from bs4.filter import SoupStrainer
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_core.documents import Document
from typing import List



def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(file_path=path)
    return loader.load()

# def load_sites(links: List[str]) -> List[Document]:
#     bs4_strainer = SoupStrainer(class_=("post-title", "post-header", "post-content"))
#     loader = WebBaseLoader(
#         web_paths=links,
#         bs_kwargs={"parse_only": bs4_strainer},
#     )
#     return loader.load()


def load_file(path: str) -> List[Document]:
    if path.endswith(".pdf"):
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
