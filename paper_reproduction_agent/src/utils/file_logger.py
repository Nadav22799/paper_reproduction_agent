"""File logger to save execution logs."""

import sys
from datetime import datetime
from pathlib import Path


class FileLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_dir: str = "./logs"):
        """Initialize file logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"execution_{timestamp}.log"

        # Open file for writing
        self.file_handle = open(self.log_file, "w", encoding="utf-8")

        self.log(f"=== Execution Log Started at {datetime.now()} ===\n")

    def log(self, message: str):
        """Write message to both console and file.

        Args:
            message: Message to log
        """
        # Print to console
        print(message, end="")

        # Write to file
        self.file_handle.write(message)
        self.file_handle.flush()  # Ensure it's written immediately

    def close(self):
        """Close the log file."""
        self.log(f"\n=== Execution Log Ended at {datetime.now()} ===")
        self.file_handle.close()
        print(f"\n📝 Full execution log saved to: {self.log_file}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()





class TeeOutput:
    """Redirect stdout to both console and file."""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        import re
        self.terminal.write(message)
        # Strip ANSI escape codes (colors, formatting) before writing to file
        clean_text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', message)
        self.log_file.write(clean_text)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()
