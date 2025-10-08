from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str = Field(..., description="The content of the document")
    metadata: dict[str, str | int | float | bool] | None = Field(default=None, description="Metadata associated with the document like source, author, etc.")