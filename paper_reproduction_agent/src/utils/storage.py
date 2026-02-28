"""
Storage Provider abstraction for the Paper Reproduction Agent.

Supports three backends controlled by the STORAGE_BACKEND env var:
  - local (default) — uses the local filesystem, identical to previous open() behaviour
  - gcs             — Google Cloud Storage (lazy import, ADC + service account)
  - s3              — AWS S3 (lazy import, standard credential chain)

Usage:
    from src.utils.storage import StorageManager

    provider = StorageManager.get_provider(base_dir="/path/to/project")
    provider.save_text("hello", "checkpoints/run1_planning.json")
    content = provider.read_text("checkpoints/run1_planning.json")
"""

from __future__ import annotations

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


# ── Abstract Base Class ────────────────────────────────────────────────────────

class StorageProvider(ABC):
    """Abstract interface for all storage backends."""

    @abstractmethod
    def save_text(self, content: str, remote_key: str) -> bool:
        """Write text content to storage at `remote_key`."""

    @abstractmethod
    def read_text(self, remote_key: str) -> Optional[str]:
        """Read text content from storage. Returns None if not found."""

    @abstractmethod
    def upload_file(self, local_path: str, remote_key: str) -> bool:
        """Upload a local file to storage."""

    @abstractmethod
    def download_file(self, remote_key: str, local_path: str) -> bool:
        """Download from storage to a local path. Returns False if not found."""

    @abstractmethod
    def list_files(self, prefix: str = "", pattern: str = "*") -> List[str]:
        """List files under `prefix` matching `pattern`. Returns relative keys."""

    @abstractmethod
    def delete(self, remote_key: str) -> bool:
        """Delete a file from storage. Returns True if deleted."""

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Return True if the key exists in storage."""


# ── Local Filesystem ───────────────────────────────────────────────────────────

class LocalStorageProvider(StorageProvider):
    """Local filesystem storage — drop-in replacement for direct open() calls."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()

    def _resolve(self, key: str) -> Path:
        return self.base_dir / key

    def save_text(self, content: str, remote_key: str) -> bool:
        dest = self._resolve(remote_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file then rename
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(dest)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def read_text(self, remote_key: str) -> Optional[str]:
        path = self._resolve(remote_key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        dest = self._resolve(remote_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return True

    def download_file(self, remote_key: str, local_path: str) -> bool:
        src = self._resolve(remote_key)
        if not src.exists():
            return False
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return True

    def list_files(self, prefix: str = "", pattern: str = "*") -> List[str]:
        search_dir = self._resolve(prefix) if prefix else self.base_dir
        if not search_dir.exists():
            return []
        return [
            str(p.relative_to(self.base_dir)).replace("\\", "/")
            for p in search_dir.glob(pattern)
            if p.is_file()
        ]

    def delete(self, remote_key: str) -> bool:
        path = self._resolve(remote_key)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, remote_key: str) -> bool:
        return self._resolve(remote_key).exists()


# ── Google Cloud Storage ───────────────────────────────────────────────────────

class GCSProvider(StorageProvider):
    """Google Cloud Storage backend with lazy import.

    Auth priority:
      1. GOOGLE_APPLICATION_CREDENTIALS env var → service account JSON
      2. Application Default Credentials (ADC) → gcloud auth / GCE metadata
    """

    def __init__(self, bucket_name: str, project_id: str, prefix: str = ""):
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS backend. "
                "Install with: pip install google-cloud-storage"
            )
        sa_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_file:
            from google.oauth2 import service_account
            _creds = service_account.Credentials.from_service_account_file(sa_file)
            self._client = gcs.Client(project=project_id, credentials=_creds)
        else:
            # Fall through to ADC
            self._client = gcs.Client(project=project_id)

        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix.rstrip("/") if prefix else ""

    def _key(self, remote_key: str) -> str:
        return f"{self._prefix}/{remote_key}" if self._prefix else remote_key

    def save_text(self, content: str, remote_key: str) -> bool:
        blob = self._bucket.blob(self._key(remote_key))
        blob.upload_from_string(content, content_type="text/plain; charset=utf-8")
        return True

    def read_text(self, remote_key: str) -> Optional[str]:
        blob = self._bucket.blob(self._key(remote_key))
        try:
            return blob.download_as_text(encoding="utf-8")
        except Exception:
            return None

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        blob = self._bucket.blob(self._key(remote_key))
        blob.upload_from_filename(local_path)
        return True

    def download_file(self, remote_key: str, local_path: str) -> bool:
        blob = self._bucket.blob(self._key(remote_key))
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "", pattern: str = "*") -> List[str]:
        import fnmatch
        full_prefix = self._key(prefix) if prefix else self._prefix
        blobs = self._client.list_blobs(self._bucket, prefix=full_prefix or None)
        result = []
        strip = (self._prefix + "/") if self._prefix else ""
        for blob in blobs:
            rel = blob.name[len(strip):] if strip and blob.name.startswith(strip) else blob.name
            if fnmatch.fnmatch(rel.split("/")[-1], pattern):
                result.append(rel)
        return result

    def delete(self, remote_key: str) -> bool:
        blob = self._bucket.blob(self._key(remote_key))
        try:
            blob.delete()
            return True
        except Exception:
            return False

    def exists(self, remote_key: str) -> bool:
        blob = self._bucket.blob(self._key(remote_key))
        return blob.exists()


# ── AWS S3 ─────────────────────────────────────────────────────────────────────

class S3Provider(StorageProvider):
    """AWS S3 backend with lazy import.

    Auth uses the standard AWS credential chain:
      env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) → ~/.aws/credentials → IAM role
    """

    def __init__(self, bucket_name: str, prefix: str = ""):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 backend. "
                "Install with: pip install boto3"
            )
        self._s3 = boto3.client("s3")
        self._bucket = bucket_name
        self._prefix = prefix.rstrip("/") if prefix else ""

    def _key(self, remote_key: str) -> str:
        return f"{self._prefix}/{remote_key}" if self._prefix else remote_key

    def save_text(self, content: str, remote_key: str) -> bool:
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(remote_key),
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return True

    def read_text(self, remote_key: str) -> Optional[str]:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._key(remote_key))
            return resp["Body"].read().decode("utf-8")
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        self._s3.upload_file(local_path, self._bucket, self._key(remote_key))
        return True

    def download_file(self, remote_key: str, local_path: str) -> bool:
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(self._bucket, self._key(remote_key), local_path)
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "", pattern: str = "*") -> List[str]:
        import fnmatch
        full_prefix = self._key(prefix) if prefix else self._prefix
        paginator = self._s3.get_paginator("list_objects_v2")
        result = []
        strip = (self._prefix + "/") if self._prefix else ""
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix or ""):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(strip):] if strip and obj["Key"].startswith(strip) else obj["Key"]
                if fnmatch.fnmatch(rel.split("/")[-1], pattern):
                    result.append(rel)
        return result

    def delete(self, remote_key: str) -> bool:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=self._key(remote_key))
            return True
        except Exception:
            return False

    def exists(self, remote_key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(remote_key))
            return True
        except Exception:
            return False


# ── Factory / Singleton ────────────────────────────────────────────────────────

class StorageManager:
    """Singleton factory that returns the configured StorageProvider.

    Reads STORAGE_BACKEND env var:
      - "local" (default) — LocalStorageProvider anchored at base_dir
      - "gcs"             — GCSProvider (needs GCP_PROJECT_ID, GCP_BUCKET_NAME)
      - "s3"              — S3Provider (needs AWS_S3_BUCKET)

    Call reset() between tests or when env vars change.
    """

    _instance: Optional[StorageProvider] = None

    @classmethod
    def get_provider(cls, base_dir: str = ".") -> StorageProvider:
        if cls._instance is not None:
            return cls._instance

        backend = os.getenv("STORAGE_BACKEND", "local").lower()

        if backend == "local":
            cls._instance = LocalStorageProvider(base_dir=base_dir)

        elif backend == "gcs":
            project = os.getenv("GCP_PROJECT_ID", "")
            bucket = os.getenv("GCP_BUCKET_NAME", "")
            if not project or not bucket:
                raise ValueError(
                    "GCS backend requires GCP_PROJECT_ID and GCP_BUCKET_NAME env vars"
                )
            cls._instance = GCSProvider(
                bucket_name=bucket,
                project_id=project,
                prefix=os.getenv("GCS_PREFIX", ""),
            )

        elif backend == "s3":
            bucket = os.getenv("AWS_S3_BUCKET", "")
            if not bucket:
                raise ValueError("S3 backend requires AWS_S3_BUCKET env var")
            cls._instance = S3Provider(
                bucket_name=bucket,
                prefix=os.getenv("AWS_S3_PREFIX", ""),
            )

        else:
            raise ValueError(
                f"Unknown STORAGE_BACKEND: '{backend}'. "
                "Choose from: local, gcs, s3"
            )

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for tests or when env vars change)."""
        cls._instance = None
