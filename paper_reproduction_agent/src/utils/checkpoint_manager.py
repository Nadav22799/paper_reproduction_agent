"""Checkpoint & Resume System for Long-Running Experiments

This module provides checkpoint functionality to save and resume experiment progress,
preventing loss of work when long experiments timeout or crash.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib


class ExperimentCheckpoint:
    """Manages checkpoints for experiment reproduction workflow."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        # Convert to absolute path for clarity
        self.checkpoint_dir = Path(checkpoint_dir).resolve()

        # Create directory and report status
        was_created = not self.checkpoint_dir.exists()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if was_created:
            print(f"📁 Created checkpoint directory: {self.checkpoint_dir}")
        else:
            # Check for existing checkpoints
            existing = list(self.checkpoint_dir.glob("*.json"))
            if existing:
                print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
                print(f"   Found {len(existing)} existing checkpoint file(s)")
            else:
                print(f"📁 Checkpoint directory: {self.checkpoint_dir} (empty)")

    def _get_checkpoint_path(self, experiment_id: str, phase: str) -> Path:
        """Get checkpoint file path for a specific experiment and phase.

        Args:
            experiment_id: Unique identifier for the experiment
            phase: Phase name (environment_setup, dataset_prep, experiment_N)

        Returns:
            Path to checkpoint file
        """
        safe_phase = phase.replace("/", "_").replace(" ", "_")
        return self.checkpoint_dir / f"{experiment_id}_{safe_phase}.json"

    def _generate_experiment_id(self, repo_path: str, paper_id: str = "") -> str:
        """Generate unique experiment ID based on repo path and paper.

        Args:
            repo_path: Path to repository
            paper_id: Optional paper identifier

        Returns:
            Unique experiment ID
        """
        content = f"{repo_path}_{paper_id}".encode()
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
            checkpoint_path = self._get_checkpoint_path(experiment_id, phase)

            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "repo_path": repo_path,
                "paper_id": paper_id,
                "state": state,
                "experiment_id": experiment_id,
            }

            # Write to temporary file first, then rename (atomic operation)
            temp_path = checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            temp_path.rename(checkpoint_path)

            # Calculate checkpoint size
            size_kb = checkpoint_path.stat().st_size / 1024
            completed_phases = state.get("completed_phases", [])

            print(f"💾 Checkpoint saved: {phase}")
            print(f"   File: {checkpoint_path.name} ({size_kb:.1f} KB)")
            print(
                f"   Completed phases: {', '.join(completed_phases) if completed_phases else 'none'}"
            )
            return True

        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
            return False

    def resume(self, repo_path: str, paper_id: str = "") -> Optional[Dict[str, Any]]:
        """Resume from last successful checkpoint.

        Args:
            repo_path: Repository path
            paper_id: Optional paper identifier

        Returns:
            Checkpoint data if found, None otherwise
        """
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)

            # Find all checkpoints for this experiment
            checkpoints = []
            for checkpoint_file in self.checkpoint_dir.glob(f"{experiment_id}_*.json"):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    checkpoints.append((checkpoint_file, data))
                except Exception:
                    continue

            # If no exact match, try flexible search by paper_id
            if not checkpoints and paper_id:
                print(
                    f"📋 No exact match for {experiment_id}, searching by paper ID..."
                )
                checkpoints = self._search_by_paper_id(paper_id)

            if not checkpoints:
                print(f"📋 No checkpoints found for experiment {experiment_id}")
                return None

            # Get most recent checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x[1]["timestamp"])
            checkpoint_path, checkpoint_data = latest_checkpoint

            print(f"📂 Resuming from checkpoint: {checkpoint_data['phase']}")
            print(f"   Saved at: {checkpoint_data['timestamp']}")
            print(f"   File: {checkpoint_path}")

            return checkpoint_data

        except Exception as e:
            print(f"⚠️  Failed to resume from checkpoint: {e}")
            return None

    def _search_by_paper_id(self, paper_id: str) -> List[tuple]:
        """Search for checkpoints by paper ID when exact experiment ID doesn't match.

        This handles cases where the repo_path is different (relative vs absolute)
        but the paper is the same.

        Args:
            paper_id: Paper identifier (arxiv ID, etc.)

        Returns:
            List of (checkpoint_path, checkpoint_data) tuples
        """
        checkpoints = []

        # Normalize paper_id for comparison
        normalized_paper_id = paper_id.replace("arxiv:", "").strip()

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check if this checkpoint matches the paper
                checkpoint_paper_id = (
                    data.get("paper_id", "").replace("arxiv:", "").strip()
                )

                if checkpoint_paper_id == normalized_paper_id:
                    checkpoints.append((checkpoint_file, data))
                    continue

                # Also check in state
                state = data.get("state", {})
                state_paper_id = (
                    state.get("paper_input", "").replace("arxiv:", "").strip()
                )

                if state_paper_id == normalized_paper_id:
                    checkpoints.append((checkpoint_file, data))
                    continue

                # Check arxiv_id in paper_metadata
                metadata = state.get("paper_metadata", {})
                arxiv_id = metadata.get("arxiv_id", "").strip()

                if arxiv_id == normalized_paper_id:
                    checkpoints.append((checkpoint_file, data))

            except Exception:
                continue

        if checkpoints:
            print(
                f"   ✅ Found {len(checkpoints)} checkpoint(s) matching paper {paper_id}"
            )

        return checkpoints

    def list_phases(self, repo_path: str, paper_id: str = "") -> List[str]:
        """List all completed phases for an experiment.

        Args:
            repo_path: Repository path
            paper_id: Optional paper identifier

        Returns:
            List of completed phase names
        """
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)

            phases = []
            for checkpoint_file in self.checkpoint_dir.glob(f"{experiment_id}_*.json"):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    phases.append(data["phase"])
                except Exception:
                    continue

            return sorted(phases)

        except Exception as e:
            print(f"⚠️  Failed to list phases: {e}")
            return []

    def clear(self, repo_path: str, paper_id: str = "") -> bool:
        """Clear all checkpoints for an experiment.

        Args:
            repo_path: Repository path
            paper_id: Optional paper identifier

        Returns:
            True if cleared successfully
        """
        try:
            experiment_id = self._generate_experiment_id(repo_path, paper_id)

            count = 0
            for checkpoint_file in self.checkpoint_dir.glob(f"{experiment_id}_*.json"):
                checkpoint_file.unlink()
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
        """Get completion status for all phases.

        Args:
            repo_path: Repository path
            paper_id: Optional paper identifier

        Returns:
            Dictionary mapping phase names to completion status
        """
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
    """Decorator to wrap functions with checkpoint save/resume logic.

    Usage:
        @create_checkpoint_aware_wrapper
        def run_experiment(state, checkpoint_manager, phase_name):
            # Your experiment code here
            return result
    """

    def wrapper(
        state,
        checkpoint_manager: ExperimentCheckpoint,
        phase_name: str,
        repo_path: str,
        paper_id: str = "",
        *args,
        **kwargs,
    ):

        # Try to resume from checkpoint
        checkpoint_data = checkpoint_manager.resume(repo_path, paper_id)

        if checkpoint_data and checkpoint_data["phase"] == phase_name:
            print(f"♻️  Resuming {phase_name} from checkpoint")
            # Merge checkpoint state with current state
            state.update(checkpoint_data["state"])
            return state

        # Run the function
        result = func(state, *args, **kwargs)

        # Save checkpoint after completion
        if result:
            checkpoint_manager.save(
                state=result, phase=phase_name, repo_path=repo_path, paper_id=paper_id
            )

        return result

    return wrapper
