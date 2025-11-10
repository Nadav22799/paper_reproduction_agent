"""Resource Detection Utility - Detect available system resources."""

import os
import subprocess
from typing import Dict


def detect_system_resources() -> Dict:
    """
    Detect available system resources (GPU, RAM, CPU).

    Returns:
        Dictionary with resource information:
        - gpu_available: bool
        - gpu_count: int
        - gpu_names: list
        - total_ram_gb: float
        - cpu_count: int
        - resource_tier: str ('high', 'medium', 'low')
    """
    resources = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "total_ram_gb": 0.0,
        "cpu_count": 0,
        "resource_tier": "low"
    }

    # Detect GPUs
    resources["gpu_available"], resources["gpu_count"], resources["gpu_names"] = _detect_gpus()

    # Detect RAM
    resources["total_ram_gb"] = _detect_ram()

    # Detect CPU cores
    resources["cpu_count"] = _detect_cpu()

    # Determine resource tier
    resources["resource_tier"] = _determine_tier(resources)

    return resources


def _detect_gpus() -> tuple:
    """Detect NVIDIA GPUs using nvidia-smi."""
    try:
        # Try nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_names = [name.strip() for name in result.stdout.strip().split("\n") if name.strip()]
            gpu_count = len(gpu_names)
            return (gpu_count > 0, gpu_count, gpu_names)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Try PyTorch as fallback
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            return (True, gpu_count, gpu_names)
    except ImportError:
        pass

    return (False, 0, [])


def _detect_ram() -> float:
    """Detect total RAM in GB."""
    try:
        # Try using psutil
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
        return round(total_ram, 2)
    except ImportError:
        pass

    # Fallback: Try reading /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    # Extract KB value and convert to GB
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 2)
    except Exception:
        pass

    # Last resort: assume moderate RAM
    return 8.0


def _detect_cpu() -> int:
    """Detect CPU core count."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _determine_tier(resources: Dict) -> str:
    """
    Determine resource tier based on available resources.

    High: Multiple GPUs OR 1 GPU + 32GB+ RAM
    Medium: 1 GPU OR 16GB+ RAM
    Low: No GPU, < 16GB RAM
    """
    gpu_count = resources["gpu_count"]
    ram_gb = resources["total_ram_gb"]

    if gpu_count >= 2 or (gpu_count == 1 and ram_gb >= 32):
        return "high"
    elif gpu_count == 1 or ram_gb >= 16:
        return "medium"
    else:
        return "low"


def get_resource_summary(resources: Dict) -> str:
    """Generate human-readable resource summary."""
    lines = []
    lines.append(f"💻 System Resources Detected:")
    lines.append(f"   GPU: {'Yes' if resources['gpu_available'] else 'No'}")

    if resources['gpu_available']:
        lines.append(f"   GPU Count: {resources['gpu_count']}")
        for i, name in enumerate(resources['gpu_names']):
            lines.append(f"      GPU {i}: {name}")

    lines.append(f"   RAM: {resources['total_ram_gb']:.1f} GB")
    lines.append(f"   CPU Cores: {resources['cpu_count']}")
    lines.append(f"   Resource Tier: {resources['resource_tier'].upper()}")

    return "\n".join(lines)


def get_experiment_strategy(resources: Dict) -> str:
    """
    Suggest experiment execution strategy based on resources.

    Returns:
        Strategy recommendation string.
    """
    tier = resources["resource_tier"]

    if tier == "high":
        return "all_experiments"  # Run all experiments from paper
    elif tier == "medium":
        return "main_experiment"  # Run main experiment with resource limits
    else:
        return "minimal_experiment"  # Run minimal/simplified version


if __name__ == "__main__":
    # Test resource detection
    resources = detect_system_resources()
    print(get_resource_summary(resources))
    print(f"\nRecommended strategy: {get_experiment_strategy(resources)}")
