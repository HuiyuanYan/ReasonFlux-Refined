from ReasonFlux.config import (
    AgentSettings,
    EmbeddingSettings,
    HierarchicalDataBaseSettings
)

from ReasonFlux.agent import BaseAgent, Navigator, Inference

from ReasonFlux.template_matcher import (
    EmbeddingService,
    OpenAIEmbeddingService,
    OllamaEmbeddingService,
    JinaAIEmbeddingService,
    HierarchicalVectorDatabase
)



def initialize_agent(config_file:str) -> BaseAgent:
    """
    Initialize an agent based on the provided configuration file.

    This function reads the agent settings from the configuration file and
    creates an instance of the specified agent type (Navigator or Inference).
    It also sets up the agent's client parameters based on the settings.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        BaseAgent: The initialized agent instance.

    Raises:
        NotImplementedError: If the specified agent type is not supported.
    """
    agent_settings:AgentSettings = AgentSettings.from_yaml(config_file)

    if agent_settings.type == "navigator":
        AgentType = Navigator
    elif agent_settings.type == "inference":
        AgentType = Inference
    else:
        raise NotImplementedError(
            f"Agent type {agent_settings.type} not supported"
        )

    agent = AgentType(
        name=agent_settings.name,
        description=agent_settings.description,
        max_steps=agent_settings.max_steps,
        client_params={
            "api_key": agent_settings.llm.api_key,
            "base_url": agent_settings.llm.base_url,
            "model": agent_settings.llm.model,
            "temperature": agent_settings.llm.temperature,
            "max_tokens": agent_settings.llm.max_tokens,
            "timeout": agent_settings.llm.timeout,
            "max_retries": agent_settings.llm.max_retries,
            "vision": agent_settings.llm.enable_vision,
            "function_calling": agent_settings.llm.enable_function_calling,
            "json_output": agent_settings.llm.enable_json_output
        }
    )

    return agent

def initialze_embedding_service(config_file: str) -> EmbeddingService:
    """
    Initialize an embedding service based on the provided configuration file.

    This function reads the embedding settings from the configuration file and
    creates an instance of the specified embedding service (OpenAI, Jina AI, or Ollama).
    It sets up the embedding function based on the provider specified in the settings.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        EmbeddingService: The initialized embedding service instance.

    Raises:
        NotImplementedError: If the specified embedding provider is not supported.
    """
    embedding_settings:EmbeddingSettings = EmbeddingSettings.from_yaml(config_file)

    match embedding_settings.provider:
        case "openai":
            embedding_service = OpenAIEmbeddingService(
                api_key=embedding_settings.api_key,
                api_base=embedding_settings.api_base,
                model=embedding_settings.model,
            )
        case "jina":
            embedding_service = JinaAIEmbeddingService(
                api_key=embedding_settings.api_key,
                model=embedding_settings.model,
            )
        case "ollama":
            embedding_service = OllamaEmbeddingService(
                url=embedding_settings.api_base,
                model_name=embedding_settings.model,
            )
        case _:
            raise NotImplementedError(
                f"Embedding provider {embedding_settings.provider} not supported"
            )
    return embedding_service

def initialize_hierarchical_database(config_file: str):
    """
    Initialize a hierarchical vector database based on the provided configuration file.

    This function reads the hierarchical database settings from the configuration file and
    creates an instance of the HierarchicalVectorDatabase. It sets up the database's data directory
    and embedding parameters based on the settings.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        HierarchicalVectorDatabase: The initialized hierarchical vector database instance.
    """
    hierarchical_settings:HierarchicalDataBaseSettings = HierarchicalDataBaseSettings.from_yaml(config_file)
    hierarchical_database = HierarchicalVectorDatabase(
        data_dir=hierarchical_settings.data_dir,
        embedding_params={
            "api_key": hierarchical_settings.embedding_service.api_key,
            "api_base": hierarchical_settings.embedding_service.api_base,
            "model": hierarchical_settings.embedding_service.model,
            "provider": hierarchical_settings.embedding_service.provider
        }
    )
    return hierarchical_database