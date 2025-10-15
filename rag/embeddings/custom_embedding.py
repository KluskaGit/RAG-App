import numpy as np
from chromadb import EmbeddingFunction, Embeddings
from huggingface_hub import InferenceClient

class CustomEmbedding(EmbeddingFunction):
    """
    Custom embedding class using Hugging Face Inference API.
    Args:
        model_name (str): The name of the model to use for generating embeddings.
        api_key (str): The API key for authenticating with the Hugging Face Inference API.
    Returns:
        List of embeddings corresponding to the input texts.
    """
    def __init__(
            self,
            model_name: str,
            api_key: str
        ):

        self.model_name = model_name
        self.api_key = api_key

        
    def __call__(self, input: list[str]) -> Embeddings:
        client = InferenceClient(
            api_key=self.api_key,
        )

        results: list = []
        for text in input:
            result = client.feature_extraction(
                text=text,
                model=self.model_name,
            )

            if isinstance(result[0], list) or isinstance(result[0], np.ndarray):
                results.append(result[0])
            else:
                results.append(result)
        return results