import json
from pydantic import BaseModel, Field, model_validator
from ReasonFlux.agent import Navigator, Inference
from ReasonFlux.template_matcher import HierarchicalVectorDatabase
from ReasonFlux.utils.client import (
    initialize_agent,
    initialize_hierarchical_database
)
from ReasonFlux.utils.common import logger
from copy import deepcopy
from typing import Dict, Any

class ReasonFlux(BaseModel):
    """
    Main class for the ReasonFlux system.

    This class integrates the Navigator agent, Inference agent, and HierarchicalVectorDatabase
    to perform reasoning tasks. It initializes the components based on configuration files
    and orchestrates the reasoning process.

    Attributes:
        navigator_config_path (str): Path to the Navigator agent configuration file.
        inference_config_path (str): Path to the Inference agent configuration file.
        hierarchical_database_config_path (str): Path to the HierarchicalVectorDatabase configuration file.
        navigator (Navigator): The Navigator agent instance.
        inference (Inference): The Inference agent instance.
        hierarchical_database (HierarchicalVectorDatabase): The HierarchicalVectorDatabase instance.
    """
    navigator_config_path: str = Field(
        default="config/navigator.yaml",
        description="The path to the navigator agent configuration file"
    )

    inference_config_path: str = Field(
        default="config/inference.yaml",
        description="The path to the inference agent configuration file"
    )

    hierarchical_database_config_path: str = Field(
        default="config/hierarchical_database.yaml",
        description="The path to the hierarchical database configuration file"
    )

    navigator: Navigator = Field(
        default=None,
        description="The navigator agent"
    )
    
    inference: Inference = Field(
        default=None,
        description="The inference agent"
    )

    hierarchical_database: HierarchicalVectorDatabase = Field(
        default=None,
        description="The hierarchical vector database"
    )


    @model_validator(mode="after")
    def initialize_reason_flux(self) -> "ReasonFlux":
        if not self.navigator or not isinstance(self.navigator, Navigator):
            self.navigator = initialize_agent(self.navigator_config_path)
        if not self.inference or not isinstance(self.inference, Inference):
            self.inference = initialize_agent(self.inference_config_path)
        if not self.hierarchical_database or not isinstance(self.hierarchical_database, HierarchicalVectorDatabase):
            self.hierarchical_database = initialize_hierarchical_database(self.hierarchical_database_config_path)
        return self
    
    def run(self, problem: str) -> Dict[str,Any] | None:
        """
        Run the ReasonFlux reasoning process for the given problem.

        This method orchestrates the entire reasoning process, including:
        1. Initializing the reasoning trajectory with the Navigator.
        2. Searching for relevant templates using the HierarchicalVectorDatabase.
        3. Dynamically adjusting the reasoning flow based on the retrieved template.
        4. Iterating through the reasoning steps using the Inference agent.

        Args:
            problem (str): The problem description to reason about.

        Returns:
            Dict[str, Any] | None: A dictionary containing metadata about the reasoning process, or None if an error occurs.
        """
        task_meta_data = {
            "problem": problem
        }
        logger.info(f"Starting ReasonFlux with problem: \n{problem}\n")

        logger.info(f"[Step1] Navigator initialize the reasoning trajectory")        
        self.navigator.initializing_reasoning_trajectory(problem)
        logger.info(f"[Step1] Navigator's reasoning thoughts: \n{self.navigator.reasoning_thoughts}\n")
        logger.info(f"[Step1] Navigator give template: \n{self.navigator.template}\n")

        task_meta_data["step1"] = {
            "reasoning_thoughts": self.navigator.reasoning_thoughts,
            "template": deepcopy(self.navigator.template),
        }

        queries = [
            self.navigator.template['General Knowledge Category'],
            self.navigator.template['Specific Direction'],
            self.navigator.template['Applied Method']
        ]

        top_k_per_level = [1, 2, 3]
        weight_per_level = [1, 0.1, 0.9]
        
        logger.info("[Step2] Hierarchical template search")
        search_result = self.hierarchical_database.hierarchical_search(
            queries=queries,
            top_k_per_level=top_k_per_level,
            weight_per_level=weight_per_level
        )

        if not search_result or not search_result[0]["meta_data"]["data"]:
            logger.error("No search result found")
            return None
        
        similarity = search_result[0]["similarity"]
        retrieved_template = json.loads(search_result[0]["meta_data"]["data"])
        logger.info(f"[Step2] Retrieved template with similarity score: {similarity}: \n{json.dumps(retrieved_template,indent=2)}\n")

        task_meta_data["step2"] = {
            "similarity": similarity,
            "template": retrieved_template
        }

        logger.info("[Step3] Navigator dynamic adjustment the reasoning flow")
        new_reasoning_flow = self.navigator.dynamic_adjustment(
            trajectory=self.navigator.reasoning_flow,
            retrieved_template=retrieved_template
        )
        logger.info(f"[Step3] New reasoning flow adjusted to: \n{new_reasoning_flow}\n")


        logger.info("[Step3] Navigator update reasoning flow")
        self.navigator.update_reasoning_flow(
            reasoning_flow_str=new_reasoning_flow
        )
        logger.info(f"[Step3] Reasoning flow updated to: \n{self.navigator.reasoning_flow}\n")

        task_meta_data["step3"] = {
            "reasoning_flow_str": new_reasoning_flow,
            "reasoning_flow": self.navigator.reasoning_flow
        }

        task_meta_data["step4"] = []

        logger.info(f"[Step4] Start reasoning process iteration")
        for step_idx in range(self.navigator.reasoning_rounds):
            current_step = self.navigator.reasoning_flow[step_idx]
            current_instruction = self.navigator.initialize_reason_problem(problem, current_step)
            logger.info(f"Iteration {step_idx + 1}/{self.navigator.reasoning_rounds} instruction: \n{current_instruction}\n")

            current_thought, current_reasoning = self.inference.interplay(
                current_instruction,
                problem,
                self.navigator.reasoning_instructions,
                self.navigator.instantiation
            )

            # Update state
            self.navigator.reasoning_instructions.append(current_instruction)
            self.navigator.instantiation.append(current_reasoning)
            logger.info(f"Iteration {step_idx + 1}/{self.navigator.reasoning_rounds} inference llm's thought: \n{current_thought}\n")
            logger.info(f"Iteration {step_idx + 1}/{self.navigator.reasoning_rounds} inference llm's reasoning: \n{current_reasoning}\n")

            task_meta_data["step4"].append(
                {
                    "instruction": current_instruction,
                    "thought": current_thought,
                    "reasoning": current_reasoning
                }
            )

        logger.info(f"[Step4] Reasoning process finished")

        return task_meta_data
    

