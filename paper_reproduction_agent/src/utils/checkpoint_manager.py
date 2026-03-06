"""Checkpoint & Resume System for Long-Running Experiments

This module provides checkpoint functionality to save and resume experiment progress,
preventing loss of work when long experiments timeout or crash.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib

from src.utils.storage import StorageProvider, StorageManager, LocalStorageProvider


class ExperimentCheckpoint:
    """Manages checkpoints for experiment reproduction workflow."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        storage: Optional[StorageProvider] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files (used as base_dir
                            for LocalStorageProvider, or as a namespace prefix).
            storage: Optional pre-configured StorageProvider. If None, a
                     LocalStorageProvider anchored at checkpoint_dir is created.
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()

        # Use the injected provider or create a local one scoped to checkpoint_dir.
        # Always create a fresh LocalStorageProvider rather than going through
        # StorageManager.get_provider(), which is a singleton and would ignore
        # the base_dir argument if a provider was already created elsewhere.
        if storage is not None:
            self._storage = storage
        else:
            self._storage = LocalStorageProvider(base_dir=str(self.checkpoint_dir))

        # Ensure local directory exists (no-op for cloud backends)
        if isinstance(self._storage, LocalStorageProvider):
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Report existing checkpoints
        existing = self._storage.list_files("", "*.json")
        if existing:
            print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
            print(f"   Found {len(existing)} existing checkpoint file(s)")
        else:
            print(f"📁 Checkpoint directory: {self.checkpoint_dir} (empty)")

    def _get_checkpoint_key(self, experiment_id: str, phase: str) -> str:
        """Get storage key for a specific experiment and phase."""
        safe_phase = phase.replace("/", "_").replace(" ", "_")
        return f"{experiment_id}_{safe_phase}.json"

    @staticmethod
    def _normalize_paper_id(paper_id: str) -> str:
        """Normalize paper ID to bare arXiv ID regardless of input format.

        Handles:
          - bare IDs:          "2309.11295" or "2309.11295v2"
          - arxiv: prefix:     "arxiv:2309.11295"
          - arXiv URLs:        "https://arxiv.org/abs/2309.11295v2"
                               "https://arxiv.org/pdf/2309.11295"
                               "https://arxiv.org/html/2309.11295v2"
        """
        if not paper_id:
            return ""
        if "arxiv.org" in paper_id:
            m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", paper_id)
            if m:
                return m.group(1)
        return paper_id.replace("arxiv:", "").strip()

    def _generate_experiment_id(self, repo_path: str, paper_id: str = "") -> str:
        """Generate unique experiment ID based on repo path and paper."""
        normalized = self._normalize_paper_id(paper_id)
        content = f"{repo_path}_{normalized}".encode()
        return hashlib.md5(content).hexdigest()[:12]

    def save(
        self, state: Dict[str, Any], phase: str, repo_path: str, paper_id: str = ""
    ) -> bool:
        """Save checkpoint for current phase.

        Args:
            state: Current state to checkpoint (must be JSON-serializable)
            phase: Current phase name
            repo_path: Repository path
            paper_id: Optional paper identifier

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)
            key = self._get_checkpoint_key(experiment_id, phase)

            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "repo_path": repo_path,
                "paper_id": paper_id,
                "state": state,
                "experiment_id": experiment_id,
            }

            content = json.dumps(checkpoint_data, indent=2)
            self._storage.save_text(content, key)

            completed_phases = state.get("completed_phases", [])
            size_kb = len(content.encode()) / 1024

            print(f"💾 Checkpoint saved: {phase}")
            print(f"   Key: {key} ({size_kb:.1f} KB)")
            print(
                f"   Completed phases: {', '.join(completed_phases) if completed_phases else 'none'}"
            )
            return True

        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
            return False

    def resume(self, repo_path: str, paper_id: str = "") -> Optional[Dict[str, Any]]:
        """Resume from last successful checkpoint."""
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)

            checkpoints = []
            for key in self._storage.list_files("", f"{experiment_id}_*.json"):
                text = self._storage.read_text(key)
                if text:
                    try:
                        data = json.loads(text)
                        checkpoints.append((key, data))
                    except Exception:
                        continue

            # If no exact match, try flexible search by paper_id
            if not checkpoints and paper_id:
                from rich import print as rprint
                rprint(
                    f"📋 [yellow]No exact match for {experiment_id}, searching by paper ID...[/yellow]"
                )
                checkpoints = self._search_by_paper_id(paper_id)

            if not checkpoints:
                print(f"📋 No checkpoints found for experiment {experiment_id}")
                return None

            # Get most recent checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x[1]["timestamp"])
            checkpoint_key, checkpoint_data = latest_checkpoint

            from rich import print as rprint
            rprint(f"📂 [bold cyan]Resuming from checkpoint:[/bold cyan] {checkpoint_data['phase']}")
            rprint(f"   Saved at: {checkpoint_data['timestamp']}")
            rprint(f"   Key: {checkpoint_key}")

            return checkpoint_data

        except Exception as e:
            import traceback
            print(f"⚠️  Failed to resume from checkpoint: {e}")
            traceback.print_exc()
            return None

    def _search_by_paper_id(self, paper_id: str) -> List[tuple]:
        """Search for checkpoints by paper ID when exact experiment ID doesn't match."""
        checkpoints = []

        # Normalize paper_id — handles bare IDs, arxiv: prefix, and full URLs
        normalized_paper_id = self._normalize_paper_id(paper_id)

        for key in self._storage.list_files("", "*.json"):
            text = self._storage.read_text(key)
            if not text:
                continue
            try:
                data = json.loads(text)

                checkpoint_paper_id = (
                    data.get("paper_id", "").replace("arxiv:", "").strip()
                )
                if checkpoint_paper_id == normalized_paper_id:
                    checkpoints.append((key, data))
                    continue

                state = data.get("state", {})
                state_paper_id = (
                    state.get("paper_input", "").replace("arxiv:", "").strip()
                )
                if state_paper_id == normalized_paper_id:
                    checkpoints.append((key, data))
                    continue

                metadata = state.get("paper_metadata", {})
                arxiv_id = metadata.get("arxiv_id", "").strip()
                if arxiv_id == normalized_paper_id:
                    checkpoints.append((key, data))

            except Exception as e:
                print(f"⚠️  Error searching checkpoint {checkpoint_file.name}: {e}")
                continue

        if checkpoints:
            print(
                f"   ✅ Found {len(checkpoints)} checkpoint(s) matching paper {paper_id}"
            )
        return checkpoints

    def list_phases(self, repo_path: str, paper_id: str = "") -> List[str]:
        """List all completed phases for an experiment."""
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)

            phases = []
            for key in self._storage.list_files("", f"{experiment_id}_*.json"):
                text = self._storage.read_text(key)
                if text:
                    try:
                        data = json.loads(text)
                        phases.append(data["phase"])
                    except Exception:
                        continue

            return sorted(phases)

        except Exception as e:
            print(f"⚠️  Failed to list phases: {e}")
            return []

    def clear(self, repo_path: str, paper_id: str = "") -> bool:
        """Clear all checkpoints for an experiment."""
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)

            count = 0
            for key in self._storage.list_files("", f"{experiment_id}_*.json"):
                self._storage.delete(key)
                count += 1

            if count > 0:
                print(
                    f"🗑️  Cleared {count} checkpoint(s) for experiment {experiment_id}"
                )
            return True

        except Exception as e:
            print(f"⚠️  Failed to clear checkpoints: {e}")
            return False

    def get_phase_status(self, repo_path: str, paper_id: str = "") -> Dict[str, bool]:
        """Get completion status for all phases."""
        phases = [
            "environment_setup",
            "dataset_preparation",
            "experiment_1",
            "experiment_2",
            "experiment_3",
        ]
        completed_phases = set(self.list_phases(repo_path, paper_id))
        return {phase: phase in completed_phases for phase in phases}


def create_checkpoint_aware_wrapper(func):
    """Decorator to wrap functions with checkpoint save/resume logic."""

    def wrapper(
        state,
        checkpoint_manager: ExperimentCheckpoint,
        phase_name: str,
        repo_path: str,
        paper_id: str = "",
        *args,
        **kwargs,
    ):
        checkpoint_data = checkpoint_manager.resume(repo_path, paper_id)

        if checkpoint_data and checkpoint_data["phase"] == phase_name:
            print(f"♻️  Resuming {phase_name} from checkpoint")
            state.update(checkpoint_data["state"])
            return state

        result = func(state, *args, **kwargs)

        if result:
            checkpoint_manager.save(
                state=result, phase=phase_name, repo_path=repo_path, paper_id=paper_id
            )

        return result

    return wrapper
