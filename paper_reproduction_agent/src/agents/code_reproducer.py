"""Code Reproducer Agent - Writes code from scratch based on paper descriptions."""

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import code_execution_tools
from ..utils.llm_factory import create_llm


class CodeReproducerState(TypedDict):
    """State for Code Reproducer Agent."""
    paper_description: str
    algorithm_pseudocode: str
    implementation_plan: str
    code_files: dict
    dependencies: list
    code_complete: bool


class CodeReproducerAgent:
    """Agent responsible for writing code from scratch."""

    def __init__(self, llm=None):
        """Initialize the Code Reproducer Agent."""
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You are an expert software engineer specialized in implementing research papers.

Your responsibilities:
1. Understand the algorithm and method from paper descriptions
2. Design a clean, modular implementation architecture
3. Write well-documented, production-quality code
4. Implement proper error handling and validation
5. Create example scripts and documentation

Implementation principles:
- Write clear, readable code with good documentation
- Follow Python best practices (PEP 8)
- Create modular, testable components
- Include type hints
- Add helpful comments for complex logic
- Create runnable examples
- Handle edge cases

When implementing:
- Start with core algorithm
- Add data loading and preprocessing
- Implement training loop (if applicable)
- Add evaluation code
- Create example usage scripts

Use the available tools to create and validate your code."""

        self.agent = create_react_agent(
            self.llm,
            tools=code_execution_tools,
            prompt=self.system_prompt
        )

    def create_implementation_plan(self, paper_analysis: dict) -> str:
        """
        Create a detailed implementation plan.

        Args:
            paper_analysis: Analysis results from PaperAnalyzerAgent

        Returns:
            Implementation plan
        """
        task = f"""Based on this paper analysis, create a detailed implementation plan:

Paper: {paper_analysis.get('paper_metadata', {})}
Algorithms: {paper_analysis.get('algorithms', [])}
Experimental Setup: {paper_analysis.get('experimental_setup', {})}

Create a plan that includes:
1. Architecture design
2. Core components to implement
3. Dependencies needed
4. File structure
5. Implementation order
6. Testing strategy

Provide a step-by-step plan."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return result.get("messages", [])[-1].content if result.get("messages") else ""

    def implement_algorithm(self, algorithm_description: str, output_dir: str = "./implementation") -> dict:
        """
        Implement an algorithm from description.

        Args:
            algorithm_description: Description or pseudocode of algorithm
            output_dir: Directory to save implementation

        Returns:
            Implementation results
        """
        task = f"""Implement this algorithm:

{algorithm_description}

Requirements:
1. Create a complete Python implementation
2. Write it to file in {output_dir}
3. Include proper documentation
4. Add type hints
5. Include example usage
6. Check syntax

Save the main implementation to {output_dir}/model.py
Create an example script at {output_dir}/example.py"""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "output_dir": output_dir,
            "result": result.get("messages", [])[-1].content if result.get("messages") else "",
        }

    def implement_training_script(self, model_description: dict, output_dir: str = "./implementation") -> str:
        """
        Create a training script.

        Args:
            model_description: Description of model and training procedure
            output_dir: Output directory

        Returns:
            Path to training script
        """
        task = f"""Create a training script for this model:

{model_description}

The script should:
1. Load and preprocess data
2. Initialize the model
3. Set up training loop
4. Include logging and checkpointing
5. Save trained model
6. Be executable

Save to {output_dir}/train.py"""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return f"{output_dir}/train.py"

    def implement_evaluation_script(self, evaluation_details: dict, output_dir: str = "./implementation") -> str:
        """
        Create an evaluation script.

        Args:
            evaluation_details: Evaluation procedure details
            output_dir: Output directory

        Returns:
            Path to evaluation script
        """
        task = f"""Create an evaluation script based on:

{evaluation_details}

The script should:
1. Load trained model
2. Load test data
3. Run evaluation
4. Compute metrics
5. Display/save results

Save to {output_dir}/evaluate.py"""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return f"{output_dir}/evaluate.py"

    def create_requirements_file(self, dependencies: list, output_path: str = "./requirements.txt") -> str:
        """
        Create requirements.txt file.

        Args:
            dependencies: List of required packages
            output_path: Output file path

        Returns:
            Status message
        """
        task = f"""Create a requirements.txt file with these dependencies:

{dependencies}

Include specific versions where important for reproducibility.
Save to {output_path}"""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return output_path
