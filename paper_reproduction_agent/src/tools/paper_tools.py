"""Tools for analyzing and extracting information from academic papers."""

import re
import arxiv
import requests
from typing import Dict, List, Any, Optional
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool
def fetch_arxiv_paper(arxiv_id: str) -> Dict[str, Any]:
    """
    Fetch paper metadata and PDF from arXiv.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345")

    Returns:
        Dictionary with paper metadata and content
    """
    try:
        import os
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        # Download PDF (create directory if it doesn't exist)
        download_dir = "./downloads"
        os.makedirs(download_dir, exist_ok=True)
        pdf_path = paper.download_pdf(dirpath=download_dir)

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)

        return {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.isoformat(),
            "arxiv_id": arxiv_id,
            "pdf_url": paper.pdf_url,
            "full_text": text,
            "categories": paper.categories,
        }
    except Exception as e:
        return {"error": f"Failed to fetch paper: {str(e)}"}


@tool
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


@tool
def extract_algorithm_pseudocode(paper_text: str) -> List[str]:
    """
    Extract algorithm pseudocode sections from paper text.

    Args:
        paper_text: Full text of the paper

    Returns:
        List of algorithm pseudocode blocks
    """
    algorithms = []

    # Pattern for algorithm environments in LaTeX
    algorithm_pattern = r'\\begin{algorithm}(.*?)\\end{algorithm}'
    matches = re.findall(algorithm_pattern, paper_text, re.DOTALL)
    algorithms.extend(matches)

    # Pattern for algorithmic environments
    algorithmic_pattern = r'\\begin{algorithmic}(.*?)\\end{algorithmic}'
    matches = re.findall(algorithmic_pattern, paper_text, re.DOTALL)
    algorithms.extend(matches)

    # Look for "Algorithm" sections
    algo_section_pattern = r'Algorithm \d+:?(.*?)(?=Algorithm \d+:|\\section|\\subsection|$)'
    matches = re.findall(algo_section_pattern, paper_text, re.DOTALL)
    algorithms.extend([m.strip() for m in matches if m.strip()])

    return algorithms if algorithms else ["No explicit algorithm pseudocode found"]


@tool
def extract_experimental_setup(paper_text: str) -> Dict[str, Any]:
    """
    Extract experimental setup details from paper.

    Args:
        paper_text: Full text of the paper

    Returns:
        Dictionary with experimental details
    """
    setup = {
        "datasets": [],
        "metrics": [],
        "hyperparameters": [],
        "baselines": [],
    }

    # Look for common section headers
    sections = {
        "experiment": r'(?:Experiments?|Experimental Setup|Evaluation)(.*?)(?=\n#|\n\\section)',
        "dataset": r'(?:Datasets?|Data)(.*?)(?=\n#|\n\\section)',
        "metric": r'(?:Metrics?|Evaluation Metrics?)(.*?)(?=\n#|\n\\section)',
    }

    for key, pattern in sections.items():
        matches = re.findall(pattern, paper_text, re.DOTALL | re.IGNORECASE)
        if matches:
            setup[f"{key}_section"] = matches[0][:1000]  # Limit size

    # Extract common dataset names
    dataset_patterns = [
        r'\b(MNIST|CIFAR-?10|CIFAR-?100|ImageNet|COCO|VOC)\b',
        r'\b(SQuAD|GLUE|SuperGLUE|WikiText)\b',
        r'\b(LibriSpeech|CommonVoice)\b',
    ]

    for pattern in dataset_patterns:
        matches = re.findall(pattern, paper_text, re.IGNORECASE)
        setup["datasets"].extend(list(set(matches)))

    # Extract metrics
    metric_patterns = [
        r'\b(accuracy|precision|recall|F1|BLEU|ROUGE|perplexity|loss)\b',
        r'\b(mAP|IoU|AUC|ROC)\b',
    ]

    for pattern in metric_patterns:
        matches = re.findall(pattern, paper_text, re.IGNORECASE)
        setup["metrics"].extend(list(set(matches)))

    return setup


@tool
def extract_results_tables(paper_text: str) -> List[str]:
    """
    Extract results tables from paper text.

    Args:
        paper_text: Full text of the paper

    Returns:
        List of table contents
    """
    tables = []

    # LaTeX table pattern
    table_pattern = r'\\begin{table}(.*?)\\end{table}'
    matches = re.findall(table_pattern, paper_text, re.DOTALL)
    tables.extend(matches)

    # Tabular pattern
    tabular_pattern = r'\\begin{tabular}(.*?)\\end{tabular}'
    matches = re.findall(tabular_pattern, paper_text, re.DOTALL)
    tables.extend(matches)

    return tables if tables else ["No tables found in standard format"]


@tool
def extract_code_references(paper_text: str) -> List[str]:
    """
    Extract code repository URLs from paper using simple search (like Ctrl+F).

    Searches for "github", "gitlab", "bitbucket" in text and extracts URLs around them.

    Args:
        paper_text: Full text of the paper

    Returns:
        List of URLs to code repositories (GitHub, GitLab, Bitbucket)
    """
    all_urls = []

    # Simple approach: Find "github" in text, then extract URL around it
    text_lower = paper_text.lower()

    # Search for each platform
    # Patterns handle URLs split across lines with optional whitespace/newlines
    platforms = [
        ('github.com', r'https?://(?:www\.)?github\.com/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)'),
        ('gitlab.com', r'https?://(?:www\.)?gitlab\.com/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)'),
        ('bitbucket.org', r'https?://(?:www\.)?bitbucket\.org/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)'),
    ]

    for platform_name, url_pattern in platforms:
        # Find all occurrences of platform name (case insensitive)
        idx = 0
        while True:
            idx = text_lower.find(platform_name, idx)
            if idx == -1:
                break

            # Extract surrounding context (500 chars before and after)
            start = max(0, idx - 500)
            end = min(len(paper_text), idx + 500)
            context = paper_text[start:end]

            # Find URLs in this context - returns tuples of (username, repo)
            matches = re.findall(url_pattern, context, re.IGNORECASE)

            # Reconstruct full URLs from captured groups
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    username, repo = match
                    # Remove any remaining whitespace/newlines
                    username = username.strip()
                    repo = repo.strip()
                    full_url = f"https://{platform_name}/{username}/{repo}"
                    all_urls.append(full_url)

            idx += len(platform_name)

    # Remove duplicates
    unique_urls = list(set(all_urls))

    return unique_urls if unique_urls else ["No code repository URLs found"]


# Tool list for easy import
paper_analysis_tools = [
    fetch_arxiv_paper,
    extract_text_from_pdf,
    extract_algorithm_pseudocode,
    extract_experimental_setup,
    extract_results_tables,
    extract_code_references,
]
