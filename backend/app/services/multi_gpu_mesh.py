"""Heterogeneous Multi-GPU Pipeline Partitioning for asymmetric legacy GPUs.

Handles partitioning transformer layers across GPUs with different VRAM capacities,
compute capabilities, and interconnect topologies (PCIe, NVLink).

Supports GPU profiles like:
- Tesla K80: 12GB VRAM, 4.5 TFLOPS FP16, PCIe 3.0 x16
- NVIDIA M40: 24GB VRAM, 11.5 TFLOPS FP16, PCIe 3.0 x16
- RTX 3060: 12GB VRAM, 16.2 TFLOPS FP16, PCIe 4.0 x16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math


@dataclass
class GPUSpec:
    """Represents a physical GPU with its capabilities."""
    id: str
    name: str
    vram_mb: int
    compute_tflops_fp16: float
    pcie_bandwidth_gbps: float
    nvlink_bandwidth_gbps: Optional[float] = None
    compute_capability: Tuple[int, int] = (0, 0)


@dataclass
class LayerProfile:
    """Profile of a transformer layer's memory footprint."""
    layer_index: int  # placeholder; actual index assigned during partitioning
    param_mb: float  # Parameter memory per layer (MB)
    activation_mb: float  # Activation memory per layer (MB)
    kv_cache_mb: float  # KV cache memory per layer (MB)
    total_mb: float  # Total per-layer VRAM requirement (MB)


@dataclass
class PartitionResult:
    """Result of heterogeneous multi-GPU layer partitioning."""
    gpu_assignments: Dict[str, List[int]]  # gpu_id -> list of layer indices
    gpu_vram_usage_mb: Dict[str, int]  # gpu_id -> used VRAM (MB)
    gpu_compute_balance: Dict[str, float]  # gpu_id -> compute TFLOPS utilization ratio
    transfer_bottlenecks_mbps: List[Tuple[str, str, float]]  # (from_gpu, to_gpu, bandwidth_mbps)
    total_batches: int
    schedule_efficiency_pct: float


# Fixed GPU profiles for common legacy asymmetric configurations
GPU_PROFILES: Dict[str, GPUSpec] = {
    "tesla_k80": GPUSpec(
        id="gpu_0",
        name="Tesla K80",
        vram_mb=12 * 1024,
        compute_tflops_fp16=4.5,
        pcie_bandwidth_gbps=16.0,  # PCIe 3.0 x16
        nvlink_bandwidth_gbps=None,  # K80 has no NVLink
        compute_capability=(3, 0),
    ),
    "m40": GPUSpec(
        id="gpu_1",
        name="NVIDIA M40",
        vram_mb=24 * 1024,
        compute_tflops_fp16=11.5,
        pcie_bandwidth_gbps=16.0,  # PCIe 3.0 x16
        nvlink_bandwidth_gbps=None,  # M40 has no NVLink
        compute_capability=(5, 2),
    ),
    "rtx_3060": GPUSpec(
        id="gpu_2",
        name="RTX 3060",
        vram_mb=12 * 1024,
        compute_tflops_fp16=16.2,
        pcie_bandwidth_gbps=32.0,  # PCIe 4.0 x16
        nvlink_bandwidth_gbps=None,
        compute_capability=(8, 6),
    ),
}


def calculate_layer_profile(
    param_billions: float,
    precision_bits: int,
    total_layers: int,
    context_len: int,
    hidden_dim: int = 8192,
) -> LayerProfile:
    """Calculate the VRAM profile per transformer layer.

    Uses the same calculation as ProfilerEngine.calculate_layer_memory but
    returns a consolidated profile.
    """
    bytes_per_param = precision_bits / 8.0
    total_param_bytes = param_billions * 1e9 * bytes_per_param
    total_param_mb = total_param_bytes / (1024 * 1024)

    # Transformer layers contain ~90% of parameters
    transformer_params_mb = total_param_mb * 0.90
    per_layer_param_mb = transformer_params_mb / max(1, total_layers)

    # Activation memory per layer (batch size = 1, FP16)
    activation_bytes = 2 * context_len * hidden_dim * 2.0
    activation_mb = activation_bytes / (1024 * 1024)

    # KV cache per layer (2 readers * 2 directions * n_heads_kv * head_dim * context_len * bytes)
    kv_cache_bytes = 2 * 2 * 8 * 128 * context_len * 2.0
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    total_mb = per_layer_param_mb + activation_mb + kv_cache_mb

    return LayerProfile(
        layer_index=0,  # placeholder; actual index assigned during partitioning
        param_mb=round(per_layer_param_mb, 2),
        activation_mb=round(activation_mb, 2),
        kv_cache_mb=round(kv_cache_mb, 2),
        total_mb=round(total_mb, 2),
    )


def assign_layers_heterogeneous(
    total_layers: int,
    layer_profile_mb: float,
    gpus: List[GPUSpec],
    prefer_nvlink: bool = True,
) -> PartitionResult:
    """Assign transformer layers across heterogeneous GPUs.

    Greedy assignment: layers are assigned to the GPU with the most remaining
    VRAM that can accommodate the layer, balancing compute capacity.

    Returns a PartitionResult with assignments, VRAM usage, compute balance,
    and transfer bottlenecks between GPUs.
    """
    if not gpus:
        raise ValueError("At least one GPU must be specified for partitioning")

    # Track remaining VRAM per GPU
    remaining_vram: Dict[str, int] = {
        gpu.id: gpu.vram_mb for gpu in gpus
    }

    # Track compute balance (inverse: lower ratio = better balanced)
    # We track TFLOPS assigned vs total available
    gpu_compute_total: Dict[str, float] = {
        gpu.id: gpu.compute_tflops_fp16 for gpu in gpus
    }
    gpu_compute_used: Dict[str, float] = {gpu.id: 0.0 for gpu in gpus}

    # Assign layers greedily
    gpu_assignments: Dict[str, List[int]] = {gpu.id: [] for gpu in gpus}

    # Sort GPUs by VRAM descending for deterministic assignment order
    sorted_gpus = sorted(gpus, key=lambda g: g.vram_mb, reverse=True)

    for layer_idx in range(total_layers):
        # Find GPUs that can fit this layer
        candidates = []
        for gpu in sorted_gpus:
            if remaining_vram[gpu.id] >= layer_profile_mb:
                # Priority: NVLink-connected GPUs first, then by compute/ VRAM ratio
                compute_efficiency = gpu.compute_tflops_fp16 / max(1, gpu.vram_mb)
                candidates.append((compute_efficiency, gpu.id))

        if not candidates:
            # Layer doesn't fit anywhere - assign to smallest GPU (will overflow)
            # This is an error condition; assign to the GPU with most remaining VRAM
            fallback_gpu = max(sorted_gpus, key=lambda g: remaining_vram[g.id])
            gpu_assignments[fallback_gpu.id].append(layer_idx)
            remaining_vram[fallback_gpu.id] -= layer_profile_mb  # allow negative (overflow)
            gpu_compute_used[fallback_gpu.id] += layer_profile_mb
            continue

        # Select the best candidate
        # Sort by: highest compute efficiency, then most VRAM
        candidates.sort(key=lambda x: (x[0],) + (0,))  # sort by compute efficiency
        best_gpu_id = candidates[0][1]

        gpu_assignments[best_gpu_id].append(layer_idx)
        remaining_vram[best_gpu_id] -= layer_profile_mb
        gpu_compute_used[best_gpu_id] += layer_profile_mb

    # Compute VRAM usage
    gpu_vram_usage_mb: Dict[str, int] = {
        gpu.id: gpu.vram_mb - remaining_vram[gpu.id] for gpu in gpus
    }

    # Compute balance: ratio of used compute to total compute per GPU
    gpu_compute_balance: Dict[str, float] = {
        gpu.id: gpu_compute_used[gpu.id] / max(1.0, gpu_compute_total[gpu.id])
        for gpu in gpus
    }

    # Compute transfer bottlenecks between GPUs
    # For PCIe-only systems, inter-GPU transfer bandwidth is PCIe bandwidth
    # For NVLink systems, use NVLink bandwidth
    transfer_bottlenecks_mbps: List[Tuple[str, str, float]] = []

    gpu_ids = [gpu.id for gpu in gpus]
    for i, from_gpu in enumerate(gpus):
        for to_gpu in gpus[i + 1:]:
            # Determine interconnect type
            from_profile = GPU_PROFILES.get(from_gpu.id, from_gpu)
            to_profile = GPU_PROFILES.get(to_gpu.id, to_gpu)

            # Use the lower of the two PCIe bandwidths as the bottleneck
            from_bw = from_profile.pcie_bandwidth_gbps or 0
            to_bw = to_profile.pcie_bandwidth_gbps or 0

            if from_profile.nvlink_bandwidth_gbps and to_profile.nvlink_bandwidth_gbps:
                bottleneck_mbps = min(from_profile.nvlink_bandwidth_gbps, to_profile.nvlink_bandwidth_gbps) * 1024
            elif from_profile.nvlink_bandwidth_gbps:
                bottleneck_mbps = from_profile.nvlink_bandwidth_gbps * 1024
            elif to_profile.nvlink_bandwidth_gbps:
                bottleneck_mbps = to_profile.nvlink_bandwidth_gbps * 1024
            else:
                # PCIe-only: use the minimum bandwidth
                bottleneck_mbps = min(from_bw, to_bw) * 1024  # GB/s to MB/s

            transfer_bottlenecks_mbps.append((from_gpu.id, to_gpu.id, bottleneck_mbps))

    # Compute schedule efficiency: how well VRAM is utilized across all GPUs
    total_vram_all = sum(gpu.vram_mb for gpu in gpus)
    total_vram_used = sum(gpu_vram_usage_mb.values())
    vram_utilization_pct = (total_vram_used / max(1, total_vram_all)) * 100.0

    # Compute total batches (1 batch = 1 round of all GPU transfers)
    # In heterogeneous setups, total batches = max batches across any GPU
    gpu_batches: Dict[str, int] = {}
    for gpu_id, layer_indices in gpu_assignments.items():
        if layer_indices:
            gpu_batches[gpu_id] = len(layer_indices)
        else:
            gpu_batches[gpu_id] = 1  # minimum 1 batch even if empty

    total_batches = max(gpu_batches.values()) if gpu_batches else 1

    schedule_efficiency_pct = min(100.0, vram_utilization_pct)

    return PartitionResult(
        gpu_assignments=gpu_assignments,
        gpu_vram_usage_mb=gpu_vram_usage_mb,
        gpu_compute_balance=gpu_compute_balance,
        transfer_bottlenecks_mbps=transfer_bottlenecks_mbps,
        total_batches=total_batches,
        schedule_efficiency_pct=round(schedule_efficiency_pct, 2),
    )


def generate_partitioning_schedule(
    total_layers: int,
    param_billions: float,
    precision_bits: int,
    context_len: int,
    hidden_dim: int = 8192,
    gpu_config_name: str = "tesla_k80+m40+rtx_3060",
) -> Dict[str, any]:
    """Generate a complete partitioning schedule for a model across a GPU config.

    Args:
        total_layers: Total number of transformer layers
        param_billions: Model parameter count in billions
        precision_bits: Quantization bits (4, 8, or 16)
        context_len: Context length
        hidden_dim: Hidden dimension size
        gpu_config_name: Dot-separated GPU names from GPU_PROFILES

    Returns:
        Dict with assignment schedule, VRAM usage, and transfer bottlenecks
    """
    # Parse GPU config
    gpu_names = [name.strip() for name in gpu_config_name.split("+")]
    gpus = [GPU_PROFILES[name] for name in gpu_names if name in GPU_PROFILES]

    if not gpus:
        raise ValueError(
            f"Unknown GPU config: {gpu_config_name}. Available: {list(GPU_PROFILES.keys())}"
        )

    # Calculate per-layer memory profile
    layer_profile = calculate_layer_profile(
        param_billions=param_billions,
        precision_bits=precision_bits,
        total_layers=total_layers,
        context_len=context_len,
        hidden_dim=hidden_dim,
    )
    layer_mb = layer_profile.total_mb

    # Assign layers across GPUs
    result = assign_layers_heterogeneous(
        total_layers=total_layers,
        layer_profile_mb=layer_mb,
        gpus=gpus,
        prefer_nvlink=True,
    )

    return {
        "layer_profile_mb": layer_mb,
        "total_layers": total_layers,
        "gpu_assignments": result.gpu_assignments,
        "gpu_vram_usage_mb": result.gpu_vram_usage_mb,
        "gpu_compute_balance": result.gpu_compute_balance,
        "transfer_bottlenecks_mbps": result.transfer_bottlenecks_mbps,
        "total_batches": result.total_batches,
        "schedule_efficiency_pct": result.schedule_efficiency_pct,
        "gpu_names": [gpu.name for gpu in gpus],
    }
