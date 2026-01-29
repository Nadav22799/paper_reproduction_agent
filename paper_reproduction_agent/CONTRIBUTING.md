# Contributing to Paper Reproduction Agent

Thank you for your interest in contributing to the **Paper Reproduction Agent**! We value your help in making this tool more robust and capable of scientifically verifying research papers.

## 🚀 Getting Started

### 1. Prerequisites
*   Python **3.10+**
*   Git

### 2. Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Nadav22799/paper_reproduction_agent.git
    cd paper_reproduction_agent
    ```

2.  **Install dependencies**:
    We recommend using a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the package in editable mode with dev dependencies
    pip install -e .[dev]
    ```

## 🛠️ Development Workflow

1.  **Create a Branch**: Always create a new branch for your work.
    ```bash
    git checkout -b feature/my-new-feature
    # or
    git checkout -b fix/bug-description
    ```

2.  **Make Changes**: Implement your feature or fix.

3.  **Code Style**:
    We use **Black** for formatting and **Ruff** for linting. Please ensure your code adheres to these standards before submitting.
    ```bash
    # Format code
    black src tests

    # Lint code
    ruff check src tests
    ```

4.  **Run Tests**:
    Ensure all tests pass.
    ```bash
    pytest
54: 
55: 5.  **Agentic Verification**:
56:     For workflow changes, run the verification set:
57:     ```bash
58:     python src/cli.py verify
59:     ```
    ```

## 📝 Pull Request Process

1.  Push your changes to your fork or branch.
2.  Open a Pull Request (PR) against the `main` branch.
3.  Provide a clear description of the problem and your solution.
4.  Link any relevant issues (e.g., "Fixes #123").
5.  Ensure CI checks pass (if applicable).

## 🐛 Reporting Issues

If you find a bug or have a feature request, please open an issue in the repository. Provide as much detail as possible:
*   Steps to reproduce the error.
*   Expected behavior vs. actual behavior.
*   Logs or screenshots.
*   The arXiv ID of the paper you were trying to reproduce (if applicable).

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License, as defined in the `LICENSE` file.
