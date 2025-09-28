from pypdf import PdfReader # Alternatively use PyMuPDF
from rag.schemas.document import Document

class FileLoader:
    
    def load_pdf(self, path: str) -> list[Document]:
        reader = PdfReader(path)
        documents: list[Document] = []
        for page in reader.pages:
            text = page.extract_text()
            documents.append(Document(text=text))

        return documents
    
    def load_file(self, path: str) -> list[Document]:
        if path.endswith(".pdf"):
            return self.load_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
