"""Dataset Manager Agent - Downloads and prepares datasets."""

from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    search_file,
    execute_shell_command,
    execute_python_script,
)
from ..utils.llm_factory import create_llm
from ..utils.message_utils import normalize_message_content


class DatasetManagerAgent:
    """Agent for downloading and preparing datasets."""

    def __init__(self, llm=None):
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You download and prepare datasets for ML experiments.

Common patterns:
- HuggingFace datasets: from datasets import load_dataset
- Torchvision: torchvision.datasets.MNIST/CIFAR10/ImageNet
- TensorFlow datasets: tensorflow_datasets.load
- Custom scripts: download_data.sh, prepare_data.py

Strategy:
1. Search README/code for dataset mentions
2. Look for existing download scripts
3. Check for dataset config files
4. Run appropriate download commands

Be pragmatic: use smallest/quickest dataset variant for testing."""

        tools = [read_file, search_file, execute_shell_command, execute_python_script]
        # Don't pass custom prompt - let ReAct use its default optimized for tool calling
        self.agent = create_react_agent(self.llm, tools=tools)

    def prepare_datasets(self, code_path: str, paper_datasets: list = None) -> Dict:
        """
        Download and prepare datasets mentioned in paper.

        Args:
            code_path: Path to repository
            paper_datasets: Dataset names from paper (optional)

        Returns:
            Dataset preparation results
        """
        dataset_info = f"Expected datasets: {paper_datasets}" if paper_datasets else "Check code for dataset requirements"

        # Prepend system context to task (since we can't override ReAct's prompt)
        task = f"""{self.system_prompt}

Task: Prepare datasets for: {code_path}

{dataset_info}

Steps:
1. Search README for dataset download instructions
2. Look for data/ or scripts/ directories with download scripts
3. Search code for dataset loading (search for 'dataset', 'load_dataset', 'download')
4. Execute download scripts if found, or note manual steps needed

Report: datasets found, download status, data location."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke({"messages": messages}, config={"recursion_limit": 30})

        return self._parse_dataset_result(result)

    def _parse_dataset_result(self, result: Dict) -> Dict:
        """Extract dataset status from agent result."""
        messages = result.get("messages", [])

        dataset_info = {
            "datasets_identified": False,
            "datasets_downloaded": False,
            "dataset_locations": [],
            "download_instructions": "",
            "errors": [],
            "report": ""
        }

        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content_str = normalize_message_content(msg.content)
                content = content_str.lower()

                if any(kw in content for kw in ["dataset", "data/", "mnist", "cifar", "imagenet", "wmt"]):
                    dataset_info["datasets_identified"] = True

                if "download" in content and ("complete" in content or "success" in content):
                    dataset_info["datasets_downloaded"] = True

                if "data/" in content or "/data" in content:
                    dataset_info["dataset_locations"].append(content_str[:100])

        # Get final report
        if messages:
            dataset_info["report"] = normalize_message_content(messages[-1].content) if hasattr(messages[-1], 'content') else ""

        return dataset_info
