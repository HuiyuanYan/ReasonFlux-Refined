from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from ReasonFlux.agent.base import BaseAgent
from ReasonFlux.prompts.inference import INTERPLAY_PROMPT
from ReasonFlux.agent.parser import think_answer_parser


class Inference(BaseAgent):
    """
    Inference agent for simulating problem-solving interactions.

    This class extends the BaseAgent and provides functionality for simulating
    the interplay between a student and a tutor. It uses a predefined prompt
    template and a model client to generate reasoning and solutions for given problems.

    Attributes:
        name (str): Name of the agent.
        description (str): Description of the agent's purpose.
    """

    name: str = "Inference"
    description: str = """Inference agent"""
    
    def step(self, chain: RunnableSerializable, **kwargs):
        return chain.invoke(kwargs)
    
    def interplay(
        self,
        instruction: str,
        problem: str,
        previous_instruction: list[str],
        previous_reasoning: list[str]
    ):
        """
        Simulates the interplay between a student and a tutor for problem-solving.

        This method constructs a conversation history based on previous instructions
        and reasoning, and then uses the model client to generate a new thought and solution.

        Args:
            instruction (str): The current instruction from the teacher.
            problem (str): The problem to be solved.
            previous_instruction (list[str]): List of previous instructions.
            previous_reasoning (list[str]): List of previous reasoning steps.

        Returns:
            tuple: A tuple containing the thought and solution generated by the model.

        Raises:
            AssertionError: If the lengths of previous_instruction and previous_reasoning do not match.
        """
        system_prompt = INTERPLAY_PROMPT

        history = []
        assert len(previous_instruction) == len(previous_reasoning), "The length of previous instruction and reasoning must be the same"
        for i in range(len(previous_instruction)):
            history.append(
                HumanMessage(content=f"Teacher Instruction for Step {i+1}: {previous_instruction[i]}")
            )
            history.append(
                AIMessage(content=previous_reasoning[i])
            )
        history.append(
            HumanMessage(content=f"Teacher Instruction for Step {len(previous_instruction)+1}: {instruction}")
        )
        
        prompt = system_prompt.__add__(
            ChatPromptTemplate.from_messages(history)
        )

        chain =  prompt | self.model_client | think_answer_parser
        res = self.run(chain, problem=problem)
        thought, solution = res["thought"], res["answer"]
        return thought, solution