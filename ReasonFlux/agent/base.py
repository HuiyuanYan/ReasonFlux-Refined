from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable
import traceback
class BaseAgent(BaseModel, ABC):
    """
    Base class for all agents, providing a common structure and functionality.

    This class defines the core attributes and methods that every agent should have.
    It includes attributes for agent identification, model client configuration,
    and execution control. It also provides a basic implementation for running
    the agent and handling exceptions.

    Attributes:
        name (str): Unique name of the agent.
        description (str): Optional description of the agent.
        model_client (ChatOpenAI): The model client used by the agent.
        max_steps (int): Maximum steps before termination.
        current_step (int): Current step in execution.
        client_params (dict): Parameters for the model client.
    """
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")
    model_client: ChatOpenAI = Field(
        None, description="The model client used by the agent"
    )
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    client_params: dict = Field(
        default={
            "api_key": "sh-xx",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-long",
            "temperature": 0.7,
            "max_tokens": 4096,
            "timeout": 60.0,
            "max_retries": 1,
            "vision": False,
            "function_calling": False,
            "json_output": False
        },
        description="Parameters for the model client",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.model_client is None or not isinstance(self.model_client, ChatOpenAI):
            self.model_client = ChatOpenAI(
                api_key=self.client_params["api_key"],
                base_url=self.client_params["base_url"],
                model=self.client_params["model"],
                temperature=self.client_params["temperature"],
                max_completion_tokens = self.client_params["max_tokens"],
                timeout=self.client_params["timeout"],
                max_retries=self.client_params["max_retries"],
                verbose=True
            )
        return self

    def run(self, chain: RunnableSerializable, **kwargs):
        """
        Run the agent's workflow.

        This method executes the agent's steps until the maximum number of steps
        is reached or an exception occurs. It handles exceptions and retries
        the step if possible.

        Args:
            chain (RunnableSerializable): The chain to run.
            **kwargs: Additional keyword arguments for the step method.

        Returns:
            Any: The result of the agent's execution, or None if it fails.
        """
        while self.current_step < self.max_steps:
            try:
                res = self.step(chain, **kwargs)
                # print(res)
                return res
            except Exception as e:
                print(f"Error in agent {self.name}: {e}")
                print(traceback.format_exc())
                if self.current_step < self.max_steps:
                    self.current_step += 1
        return None

    @abstractmethod
    def step(self, messages:dict, **kwargs):
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """