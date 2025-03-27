from ReasonFlux.agent.parser.think_answer_parser import ThinkAnswerOutputParser
from langchain_core.output_parsers import JsonOutputParser

think_answer_parser = ThinkAnswerOutputParser()
json_parser = JsonOutputParser()