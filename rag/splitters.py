import tiktoken

class TokenTextSplitter:
    def __init__(
            self, chunk_size: int=300,
            overlap: int=50,
            tokenizer: str='cl100k_base',
            ):
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc= tiktoken.get_encoding(tokenizer)

    def split_text(self, text: str) -> list[str]:
        tokens = self.enc.encode(text)

        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk = self.enc.decode(tokens[start:end])
            chunks.append(chunk)
            start = end - self.overlap

        return chunks

