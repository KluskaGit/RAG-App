from pydantic import BaseModel, Field
from typing import Any


class Document(BaseModel):
    text: str = Field(..., description="The content of the document")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata associated with the document like source, author, etc.")