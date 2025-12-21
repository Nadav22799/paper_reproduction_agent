"""Out-of-Memory (OOM) Handler for Adaptive Batch Size Adjustment

This module detects OOM errors and automatically adjusts batch sizes to fit
available GPU memory, preventing experiment failures.
"""

import re
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class OOMHandler:
    """Handles out-of-memory errors by adjusting batch sizes."""

    def __init__(self):
        """Initialize OOM handler."""
        self.gpu_memory_gb = self._detect_gpu_memory()
        self.oom_history = []  # Track OOM adjustments
        self.batch_size_reduction_factor = 0.5  # Halve batch size on OOM

    def _detect_gpu_memory(self) -> List[float]:
        """Detect available GPU memory for each GPU.

        Returns:
            List of GPU memory sizes in GB, empty list if no GPUs
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                memory_mb = [float(line.strip()) for line in result.stdout.strip().split('\n')]
                memory_gb = [mb / 1024 for mb in memory_mb]
                print(f"🎮 Detected {len(memory_gb)} GPU(s): {[f'{gb:.1f}GB' for gb in memory_gb]}")
                return memory_gb

        except Exception as e:
            print(f"⚠️  Could not detect GPU memory: {e}")

        return []

    def detect_oom_error(self, output: str) -> bool:
        """Detect if output contains OOM error.

        Args:
            output: Command output (stdout + stderr)

        Returns:
            True if OOM error detected
        """
        oom_patterns = [
            r'CUDA out of memory',
            r'OutOfMemoryError',
            r'RuntimeError.*out of memory',
            r'CUDA error.*out of memory',
            r'torch\.cuda\.OutOfMemoryError',
            r'allocation failed',
            r'cudaMalloc failed',
        ]

        output_lower = output.lower()
        return any(re.search(pattern, output, re.IGNORECASE) for pattern in oom_patterns)

    def extract_batch_size_params(self, script_path: str) -> Dict[str, Any]:
        """Extract batch size parameters from script.

        Args:
            script_path: Path to script file (.sh or .py)

        Returns:
            Dictionary with batch size parameter info:
                - patterns: List of (param_name, current_value) tuples
                - script_type: 'shell' or 'python'
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            patterns = []

            # Common batch size parameter patterns
            batch_param_patterns = [
                # Python argument patterns
                (r'--batch[_-]?size["\']?\s*[=:]\s*(\d+)', 'batch_size'),
                (r'--per[_-]?device[_-]?train[_-]?batch[_-]?size["\']?\s*[=:]\s*(\d+)', 'per_device_train_batch_size'),
                (r'--per[_-]?device[_-]?eval[_-]?batch[_-]?size["\']?\s*[=:]\s*(\d+)', 'per_device_eval_batch_size'),
                (r'--train[_-]?batch[_-]?size["\']?\s*[=:]\s*(\d+)', 'train_batch_size'),
                (r'--eval[_-]?batch[_-]?size["\']?\s*[=:]\s*(\d+)', 'eval_batch_size'),

                # Shell variable patterns
                (r'BATCH[_-]?SIZE\s*=\s*(\d+)', 'BATCH_SIZE'),
                (r'batch[_-]?size\s*=\s*(\d+)', 'batch_size'),

                # Python code patterns
                (r'batch_size\s*=\s*(\d+)', 'batch_size'),
                (r'train_batch_size\s*=\s*(\d+)', 'train_batch_size'),
            ]

            for pattern, param_name in batch_param_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    current_value = int(match.group(1))
                    patterns.append({
                        'param_name': param_name,
                        'current_value': current_value,
                        'pattern': pattern,
                        'full_match': match.group(0)
                    })

            script_type = 'shell' if script_path.endswith('.sh') else 'python'

            return {
                'patterns': patterns,
                'script_type': script_type,
                'has_batch_params': len(patterns) > 0
            }

        except Exception as e:
            print(f"⚠️  Failed to extract batch size params: {e}")
            return {'patterns': [], 'script_type': 'unknown', 'has_batch_params': False}

    def adjust_batch_size_in_script(self, script_path: str, reduction_factor: float = 0.5) -> Tuple[bool, Dict[str, Any]]:
        """Adjust batch sizes in script file.

        Args:
            script_path: Path to script file
            reduction_factor: Factor to reduce batch size by (default: 0.5 = half)

        Returns:
            Tuple of (success, adjustment_info)
        """
        try:
            # Read script
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Extract batch size parameters
            param_info = self.extract_batch_size_params(script_path)

            if not param_info['has_batch_params']:
                print(f"⚠️  No batch size parameters found in {script_path}")
                return False, {'reason': 'no_batch_params'}

            adjustments = []

            # Adjust each batch size parameter
            for param in param_info['patterns']:
                old_value = param['current_value']
                new_value = max(1, int(old_value * reduction_factor))

                # Replace in content
                # Need to be careful to match the exact pattern
                old_match = param['full_match']
                new_match = old_match.replace(str(old_value), str(new_value))

                content = content.replace(old_match, new_match, 1)  # Replace first occurrence only

                adjustments.append({
                    'param': param['param_name'],
                    'old': old_value,
                    'new': new_value
                })

                print(f"   📉 {param['param_name']}: {old_value} → {new_value}")

            # Only write if changes were made
            if content != original_content:
                # Backup original file
                backup_path = Path(script_path).with_suffix(Path(script_path).suffix + '.bak')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                # Write adjusted script
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"💾 Script adjusted, backup saved to {backup_path}")

                return True, {
                    'adjustments': adjustments,
                    'backup_path': str(backup_path),
                    'reduction_factor': reduction_factor
                }

            return False, {'reason': 'no_changes_needed'}

        except Exception as e:
            print(f"❌ Failed to adjust batch sizes: {e}")
            return False, {'reason': 'error', 'error': str(e)}

    def suggest_batch_size(self, current_batch_size: int, gpu_memory_gb: float,
                          model_size: str = "unknown") -> int:
        """Suggest appropriate batch size based on GPU memory.

        Args:
            current_batch_size: Current batch size that caused OOM
            gpu_memory_gb: GPU memory in GB
            model_size: Model size category ('small', 'medium', 'large', 'unknown')

        Returns:
            Suggested batch size
        """
        # Heuristics based on typical memory usage
        # These are rough estimates and may need tuning
        memory_rules = {
            'small': {  # < 100M parameters (BERT-base, GPT2-small)
                8: 32,   # 8GB GPU → batch 32
                16: 64,  # 16GB GPU → batch 64
                24: 96,  # 24GB GPU → batch 96
                40: 128, # 40GB GPU → batch 128
                48: 160, # 48GB GPU → batch 160
            },
            'medium': {  # 100M-1B parameters (BERT-large, GPT2-medium)
                8: 16,
                16: 32,
                24: 48,
                40: 64,
                48: 96,
            },
            'large': {  # 1B+ parameters (GPT2-large, GPT-3)
                8: 4,
                16: 8,
                24: 16,
                40: 32,
                48: 48,
            },
            'unknown': {  # Conservative defaults
                8: 16,
                16: 32,
                24: 48,
                40: 64,
                48: 96,
            }
        }

        # Get closest memory tier
        memory_tiers = [8, 16, 24, 40, 48]
        closest_tier = min(memory_tiers, key=lambda x: abs(x - gpu_memory_gb))

        suggested = memory_rules.get(model_size, memory_rules['unknown']).get(closest_tier, 8)

        # If current batch size is known and caused OOM, suggest half
        if current_batch_size > 0:
            suggested = min(suggested, max(1, int(current_batch_size * self.batch_size_reduction_factor)))

        return suggested

    def handle_oom(self, script_path: str, error_output: str,
                   attempt: int = 1, max_attempts: int = 3) -> Dict[str, Any]:
        """Handle OOM error by adjusting batch size and suggesting retry.

        Args:
            script_path: Path to script that caused OOM
            error_output: Error output containing OOM message
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum retry attempts

        Returns:
            Dictionary with handling result:
                - should_retry: Whether to retry
                - adjusted: Whether script was adjusted
                - message: Human-readable message
                - adjustment_info: Details of adjustments made
        """
        if attempt > max_attempts:
            return {
                'should_retry': False,
                'adjusted': False,
                'message': f"Max OOM retry attempts ({max_attempts}) reached",
                'adjustment_info': {}
            }

        print(f"\n🔥 OOM Error Detected (Attempt {attempt}/{max_attempts})")
        print(f"   Script: {script_path}")

        # Adjust batch size
        success, adjustment_info = self.adjust_batch_size_in_script(
            script_path,
            reduction_factor=self.batch_size_reduction_factor ** attempt  # Reduce more on each attempt
        )

        if success:
            self.oom_history.append({
                'script': script_path,
                'attempt': attempt,
                'adjustments': adjustment_info.get('adjustments', [])
            })

            return {
                'should_retry': True,
                'adjusted': True,
                'message': f"Batch size adjusted (attempt {attempt}). Retrying...",
                'adjustment_info': adjustment_info
            }
        else:
            reason = adjustment_info.get('reason', 'unknown')

            if reason == 'no_batch_params':
                # Try to add batch size parameter to command
                return {
                    'should_retry': False,
                    'adjusted': False,
                    'message': "No batch size parameters found. Consider adding --batch_size argument.",
                    'adjustment_info': adjustment_info,
                    'suggestion': "Add --batch_size 8 to the command line"
                }
            else:
                return {
                    'should_retry': False,
                    'adjusted': False,
                    'message': f"Could not adjust batch size: {reason}",
                    'adjustment_info': adjustment_info
                }

    def restore_original_script(self, script_path: str) -> bool:
        """Restore original script from backup.

        Args:
            script_path: Path to script file

        Returns:
            True if restored successfully
        """
        try:
            backup_path = Path(script_path).with_suffix(Path(script_path).suffix + '.bak')

            if backup_path.exists():
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                print(f"♻️  Restored original script from {backup_path}")
                return True
            else:
                print(f"⚠️  No backup found at {backup_path}")
                return False

        except Exception as e:
            print(f"❌ Failed to restore script: {e}")
            return False

    def get_oom_summary(self) -> str:
        """Get summary of OOM adjustments made during session.

        Returns:
            Human-readable summary
        """
        if not self.oom_history:
            return "No OOM errors encountered."

        lines = ["OOM Handling Summary:", ""]

        for i, event in enumerate(self.oom_history, 1):
            lines.append(f"{i}. Script: {event['script']} (Attempt {event['attempt']})")
            for adj in event['adjustments']:
                lines.append(f"   - {adj['param']}: {adj['old']} → {adj['new']}")

        return "\n".join(lines)
