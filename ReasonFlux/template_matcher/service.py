from abc import ABC
from pydantic import BaseModel, Field
import numpy as np
from chromadb.utils import embedding_functions

class EmbeddingService(BaseModel, ABC):
    """
    Abstract base class for embedding services.

    This class provides a common interface for different embedding services.
    It defines the basic structure and methods that should be implemented by
    any concrete embedding service.

    Attributes:
        embedding_function (embedding_functions.EmbeddingFunction): The embedding function used by the embedding service.
    """
    embedding_function: embedding_functions.EmbeddingFunction = Field(
        default=None,
        description="The embedding function used by the embedding service"
    )
    class Config:
        arbitrary_types_allowed: bool = True

    def encode(self, text: str) -> np.ndarray:
        return np.array(self.embedding_function([text])[0])


class OpenAIEmbeddingService(EmbeddingService):
    """
    Implementation of the embedding service using OpenAI's embedding model.

    This class provides a concrete implementation of the embedding service using
    OpenAI's embedding model. It initializes the embedding function with the
    provided API key, base URL, and model name.

    Attributes:
        api_key (str): The API key for OpenAI.
        api_base (str): The base URL for OpenAI.
        model_name (str): The name of the OpenAI embedding model.
    """
    def __init__(self,
        api_key:str,
        api_base:str,
        model_name: str = "text-embedding-v3"
    ):
        super().__init__()
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key = api_key,
            api_base = api_base,
            model_name=model_name
        )

class JinaAIEmbeddingService(EmbeddingService):
    """
    Implementation of the embedding service using Jina AI's embedding model.

    This class provides a concrete implementation of the embedding service using
    Jina AI's embedding model. It initializes the embedding function with the
    provided API key and model name.

    Attributes:
        api_key (str): The API key for Jina AI.
        model_name (str): The name of the Jina AI embedding model.
    """
    def __init__(
        self,
        api_key:str,
        model_name: str = "jinaai/jina-embeddings-v3"
    ):
        super().__init__()
        self.embedding_function = embedding_functions.JinaEmbeddingFunction(
            model_name=model_name,
            api_key=api_key
        )

class OllamaEmbeddingService(EmbeddingService):
    """
    Implementation of the embedding service using Ollama's embedding model.

    This class provides a concrete implementation of the embedding service using
    Ollama's embedding model. It initializes the embedding function with the
    provided URL and model name.

    Attributes:
        url (str): The URL for Ollama.
        model_name (str): The name of the Ollama embedding model.
    """
    def __init__(
        self,
        url:str,
        model_name: str = "llama2"
    ):
        super().__init__()
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            url=url,
            model_name=model_name
        )