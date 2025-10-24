import os
import chromadb.utils.embedding_functions as ef
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import Union

# Mapping of provider names to their corresponding embedding function classes
EMBEDDING_PROVIDERS: dict[str, type] = {
    "openai": ef.OpenAIEmbeddingFunction,
    "huggingface": ef.HuggingFaceEmbeddingFunction,
    "ollama": ef.OllamaEmbeddingFunction,
    "cohere": ef.CohereEmbeddingFunction,
    "google_palm": ef.GooglePalmEmbeddingFunction,
    "google_vertex": ef.GoogleVertexEmbeddingFunction,
    "google_generative_ai": ef.GoogleGenerativeAiEmbeddingFunction,
    "instructor": ef.InstructorEmbeddingFunction,
    "jina": ef.JinaEmbeddingFunction,
    "sentence_transformer": ef.SentenceTransformerEmbeddingFunction,
    "amazon_bedrock": ef.AmazonBedrockEmbeddingFunction,
    "mistral": ef.MistralEmbeddingFunction,
    "voyageai": ef.VoyageAIEmbeddingFunction,
    "together_ai": ef.TogetherAIEmbeddingFunction,
    "roboflow": ef.RoboflowEmbeddingFunction,
    "text2vec": ef.Text2VecEmbeddingFunction,
    "openclip": ef.OpenCLIPEmbeddingFunction,
    "morph": ef.MorphEmbeddingFunction,
    "cloudflare_workers_ai": ef.CloudflareWorkersAIEmbeddingFunction,
    "baseten": ef.BasetenEmbeddingFunction,
    "bm25": ef.Bm25EmbeddingFunction,
    "huggingface_sparse": ef.HuggingFaceSparseEmbeddingFunction,
    "fastembed_sparse": ef.FastembedSparseEmbeddingFunction,
    "chroma_langchain": ef.ChromaLangchainEmbeddingFunction,
    "default": ef.DefaultEmbeddingFunction,
}


def get_embedding_function(**kwargs) -> Union[EmbeddingFunction, object]:
    """
    Get the embedding function based on the provider name.
    
    Available providers: openai, huggingface, ollama, cohere, google_palm, 
    google_vertex, google_generative_ai, instructor, jina, sentence_transformer, 
    amazon_bedrock, mistral, voyageai, together_ai, roboflow, text2vec, 
    openclip, morph, cloudflare_workers_ai, baseten, bm25, huggingface_sparse,
    fastembed_sparse, chroma_langchain, default.
    
    Args:
        **kwargs: Additional keyword arguments for the embedding function.
    
    Returns:
        Union[EmbeddingFunction, object]: The corresponding embedding function instance.
    
    Raises:
        ValueError: If the provider argument is missing or not recognized.
    """
    try:
        provider = kwargs.pop("provider").lower()
    except KeyError:
        raise ValueError("The 'provider' argument is required.")
    
    if "api_key" in kwargs:
        try:
            kwargs["api_key"] = os.environ[kwargs["api_key"]]
        except Exception as e:
            raise ValueError(f"Error retrieving API key from environment: {e}")

    
    embedding_class = EMBEDDING_PROVIDERS.get(provider)
    
    if embedding_class is None:
        available_providers = ", ".join(sorted(EMBEDDING_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available providers: {available_providers}"
        )
    
    return embedding_class(**kwargs)