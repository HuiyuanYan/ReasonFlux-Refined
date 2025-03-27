import re
from typing import Any, Dict

from langchain.schema import BaseOutputParser, OutputParserException


# Custom instructions for parsing <think></think> tags.
# If <think></think> tags are not found, "thought" will be set to an empty string,
# and "answer" will be set to the entire text.
THINK_INSTRUCTIONS = """
The output should be formatted as HTML-like tags.
Here is the output example:
<think>
think content
</think>
<answer>
answer content
</answer>
Note: If <think></think> tags are not found, "thought" will be set to an empty string,
and "answer" will be set to the entire text.
"""


class ThinkAnswerOutputParser(BaseOutputParser[Dict[str, str]]):
    """
    Custom parser to extract content from <think></think> tags.
    If <think></think> tags are not found, "thought" will be set to an empty string,
    and "answer" will be set to the entire text.
    """

    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse the model's output to extract content within <think> tags.
        If <think> tags are found, everything after </think> is considered as the answer.
        If <think> tags are not found, "thought" is set to an empty string, and "answer" is set to the entire text.
        """
        # Define regex pattern for <think> tags
        think_pattern = r"<think>(.*?)</think>"

        # Perform search for <think> tags
        think_match = re.search(think_pattern, text, re.DOTALL)

        # Check if <think> tags are found
        if think_match:
            # Extract content within <think> tags
            think_content = think_match.group(1).strip()
            # Extract everything after </think> as the answer
            answer_content = text[think_match.end():].strip()
        else:
            # If <think> tags are not found, set "thought" to an empty string and "answer" to the entire text
            think_content = ""
            answer_content = text.strip()

        return {"thought": think_content, "answer": answer_content}

    def get_format_instructions(self) -> str:
        """
        Provide formatting instructions for the model's output.
        """
        return THINK_INSTRUCTIONS

    @property
    def _type(self) -> str:
        """
        Return the type of this parser.
        """
        return "ThinkAnswerParser"


if __name__ == "__main__":
    parser = ThinkAnswerOutputParser()
    text_with_think = "<think>Thinking content</think>Answer content"
    text_without_think = "No think tags in this text"
    
    result_with_think = parser.parse(text_with_think)
    result_without_think = parser.parse(text_without_think)

    print("Result with <think> tags:", result_with_think)
    print("Result without <think> tags:", result_without_think)