from pydantic import BaseModel, Field
from pydantic_yaml import parse_yaml_file_as
from typing import Literal

class YamlSettings(BaseModel):
    @classmethod
    def from_yaml(cls, yaml_file: str) -> "YamlSettings":
        """
        Loads the LLM settings from a YAML file."
        """
        return parse_yaml_file_as(cls, yaml_file)


class LLMSettings(YamlSettings):
    model: str = Field(..., description="Model Name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    temperature: float = Field(1.0, description="Sampling temperature")
    timeout:float = Field(60.0, description="Timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    
    enable_vision:bool = Field(False, description="Enable vision")
    enable_function_calling:bool = Field(False, description="Enable function calling")
    enable_json_output:bool = Field(False, description="Enable JSON output")

class AgentSettings(YamlSettings):
    name: str = Field(..., description="Unique name of the agent")
    description: str = Field(..., description="Description of the agent")
    type: Literal["inference", "navigator"] = Field(..., description="Type of agent")
    max_steps: int = Field(10, description="Maximum number of steps")
    llm: LLMSettings = Field(..., description="LLM settings")

class EmbeddingSettings(YamlSettings):
    model: str = Field(..., description="Model Name")
    api_key: str = Field(..., description="API key")
    api_base: str = Field(..., description="API base URL")
    provider: Literal["openai", "jina"] = Field(..., description="Embedding provider")

class HierarchicalDataBaseSettings(YamlSettings):
    data_dir: str = Field(..., description="Data directory")
    embedding_service: EmbeddingSettings = Field(..., description="Embedding service")