from bs4 import SoupStrainer
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_core.documents import Document


def load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(file_path=path)
    return loader.load()

# def load_sites(links: list[str]) -> list[Document]:
#     bs4_strainer = SoupStrainer(class_=("post-title", "post-header", "post-content"))
#     loader = WebBaseLoader(
#         web_paths=links,
#         bs_kwargs={"parse_only": bs4_strainer},
#     )
#     return loader.load()


def load_file(path: str) -> list[Document]:
    if path.endswith(".pdf"):
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
