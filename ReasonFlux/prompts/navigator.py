from langchain_core.prompts import ChatPromptTemplate

TRAJECTORY_BUILDING_PROMPT = ChatPromptTemplate(
    [
        ("system", "Please construct a reasoning trajectory in JSON format based on the following problem. This reasoning trajectory should include the following: problem description, general knowledge category, specific direction, applied method, examined knowledge, and reason_flow. Please output according to the given format:\n{{\"Problem\": Here describes the problem you constructed, \"General Knowledge Category\": Here corresponds to the general category of mathematical knowledge to which the problem belongs, \"Specific Direction\": Here corresponds to the specific knowledge direction to which the problem belongs, \"Applied Method\": Here corresponds to the `template_name` of the input template, **please use its original name completely, do not refer, abbreviate or rewrite**, \"Examined Knowledge\": [Here is a list used to list the knowledge tags examined by this problem], \"reason_flow\": [This is a list, according to the `reason_flow` steps in the input template, to describe in detail the thinking process of solving the problem. Each step should be explained in conjunction with the specific situation of the problem, such as how to convert conditions, how to apply formulas, etc. But it should be noted that `reason_flow` is only a framework to guide students' thinking, and cannot directly give specific calculation results or answers, but should retain a certain degree of challenge for students to complete the specific calculations and derivations themselves.]}}. Before providing a formal response, please carefully consider and analyze the question, and place your thoughts within <think></think> tags."),
        ("user", "{problem}")
    ]
)

TRAJECTORY_ADJUST_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """As a math problem-solving tutor, you need to optimize the original reasoning flow based on two inputs:
1. The user-provided **Original Reason Flow**
2. The retrieved **Standard Solution Template**

Original Reason Flow:
{original_reason_flow}

Standard Solution Template:
{standard_solution_template}

Perform the following optimizations:
① **Step Consolidation**: Merge similar operations (e.g., multiple calculations → "Generate initial data")
② **Pattern Abstraction**: Convert concrete observations into methodological prompts (e.g., "Identify relationships with exponential terms")
③ **Logic Formalization**: Mark critical nodes using standard mathematical induction phases (base case/assumption/recursion)
④ **Gap Preservation**: Replace numerical computations with placeholders (e.g., "Analyze residual patterns")

Output Requirements:
- Maintain the logical framework from the standard template
- Each step contains only **1 core thinking instruction**
- Use methodological verbs (Observe/Hypothesize/Verify/Derive)
- Prohibit specific numerical values or algebraic operations
please carefully consider and analyze the question, and place your thoughts within <think></think> tags.
"""
        )
    ]
)

# here we make slight modifications to the prompt to make llm's output conform to the general format of a JSON list
REASONING_FLOW_UPDATE_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """Please extract the reasoning flows from the following text and output in JSON list format, each element in the JSON list should be a step in the reasoning process. Ensure that any special characters, such as backslashes (`\\`), are properly escaped to produce a valid JSON string:

Input Reasoning Flow:
{reasoning_flow}

Note that only output the JSON list of reasoning flow in your response."""
        )
    ]
)

INITIALIZE_REASON_PROBLEM_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """You are a math tutor guiding a student to solve a math problem based on the given step.
Your task is to help your student to learn how to apply the steps to solve the problem.
Based on the problem description and the current step, give a clear and high-level instruction for your student to help them apply the method in the current step to solve the problem.

Problem:
{problem}"""
        )
    ]
)