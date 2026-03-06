"""File logger to save execution logs."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.storage import StorageProvider


class FileLogger:
    """Logger that writes to both console and file.

    Streaming writes always go to a local file for performance. If a cloud
    StorageProvider is supplied, `finalize()` (called by `close()`) uploads
    the completed log file to the configured backend.
    """

    def __init__(self, log_dir: str = "./logs", storage: Optional[StorageProvider] = None):
        """Initialize file logger.

        Args:
            log_dir: Directory to store log files locally.
            storage: Optional StorageProvider for cloud upload on close.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self._storage = storage

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"execution_{timestamp}.log"

        # Open file for writing (streaming append)
        self.file_handle = open(self.log_file, "w", encoding="utf-8")

        self.log(f"=== Execution Log Started at {datetime.now()} ===\n")

    def log(self, message: str):
        """Write message to both console and file."""
        print(message, end="")
        self.file_handle.write(message)
        self.file_handle.flush()

    def finalize(self):
        """Upload the completed log file to cloud storage if configured."""
        if self._storage is None:
            return
        from src.utils.storage import LocalStorageProvider
        if isinstance(self._storage, LocalStorageProvider):
            return  # Already on local disk, nothing to upload
        try:
            remote_key = f"logs/{self.log_file.name}"
            self._storage.upload_file(str(self.log_file), remote_key)
            print(f"☁️  Log uploaded to storage: {remote_key}")
        except Exception as e:
            print(f"⚠️  Failed to upload log to cloud storage: {e}")

    def close(self):
        """Close the log file and upload to cloud if configured."""
        self.log(f"\n=== Execution Log Ended at {datetime.now()} ===")
        self.file_handle.close()
        print(f"\n📝 Full execution log saved to: {self.log_file}")
        self.finalize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TeeOutput:
    """Redirect stdout to both console and file.

    If a StorageProvider is supplied, the completed log is uploaded on close().
    """

    def __init__(self, log_file, storage: Optional[StorageProvider] = None):
        self.terminal = sys.stdout
        self.log_path = Path(log_file)
        self.log_file = open(log_file, "w", encoding="utf-8")
        self._storage = storage

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
        # Upload to cloud if configured
        if self._storage is not None:
            from src.utils.storage import LocalStorageProvider
            if not isinstance(self._storage, LocalStorageProvider):
                try:
                    remote_key = f"logs/{self.log_path.name}"
                    self._storage.upload_file(str(self.log_path), remote_key)
                    self.terminal.write(f"☁️  Log uploaded to storage: {remote_key}\n")
                except Exception as e:
                    self.terminal.write(f"⚠️  Failed to upload log: {e}\n")

    def isatty(self):
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()
