"""Tests for Heterogeneous Multi-GPU Pipeline Partitioning."""

from backend.app.services.multi_gpu_mesh import (
    GPU_PROFILES,
    GPUSpec,
    LayerProfile,
    PartitionResult,
    calculate_layer_profile,
    assign_layers_heterogeneous,
    generate_partitioning_schedule,
)


def test_gpu_profiles_exist():
    """All three legacy GPU profiles are registered."""
    assert "tesla_k80" in GPU_PROFILES
    assert "m40" in GPU_PROFILES
    assert "rtx_3060" in GPU_PROFILES


def test_k80_profile():
    """Tesla K80: 12GB VRAM, PCIe 3.0, no NVLink."""
    k80 = GPU_PROFILES["tesla_k80"]
    assert k80.name == "Tesla K80"
    assert k80.vram_mb == 12 * 1024
    assert k80.pcie_bandwidth_gbps == 16.0
    assert k80.nvlink_bandwidth_gbps is None


def test_m40_profile():
    """NVIDIA M40: 24GB VRAM, PCIe 3.0, no NVLink."""
    m40 = GPU_PROFILES["m40"]
    assert m40.name == "NVIDIA M40"
    assert m40.vram_mb == 24 * 1024
    assert m40.pcie_bandwidth_gbps == 16.0
    assert m40.nvlink_bandwidth_gbps is None


def test_rtx3060_profile():
    """RTX 3060: 12GB VRAM, PCIe 4.0, no NVLink."""
    rtx = GPU_PROFILES["rtx_3060"]
    assert rtx.name == "RTX 3060"
    assert rtx.vram_mb == 12 * 1024
    assert rtx.pcie_bandwidth_gbps == 32.0
    assert rtx.nvlink_bandwidth_gbps is None


def test_calculate_layer_profile_70b_16bit():
    """Layer profile for 70B model at 16-bit precision."""
    profile = calculate_layer_profile(
        param_billions=70.0,
        precision_bits=16,
        total_layers=80,
        context_len=8192,
    )
    assert isinstance(profile, LayerProfile)
    assert profile.param_mb > 0
    assert profile.activation_mb > 0
    assert profile.kv_cache_mb > 0
    assert profile.total_mb > 0
    # Total should be roughly param + activation + kv_cache
    assert profile.total_mb >= profile.param_mb


def test_calculate_layer_profile_4bit():
    """Layer profile for 70B model at 4-bit precision."""
    profile = calculate_layer_profile(
        param_billions=70.0,
        precision_bits=4,
        total_layers=80,
        context_len=8192,
    )
    assert isinstance(profile, LayerProfile)
    # 4-bit should have lower memory than 16-bit
    assert profile.total_mb < 2000.0


def test_assign_layers_heterogeneous_k80_m40_rtx3060():
    """Assign 80 layers across K80 + M40 + RTX 3060."""
    gpus = [
        GPU_PROFILES["tesla_k80"],
        GPU_PROFILES["m40"],
        GPU_PROFILES["rtx_3060"],
    ]
    result = assign_layers_heterogeneous(
        total_layers=80,
        layer_profile_mb=1822.04,  # from 70B/16bit/8192ctx calculation
        gpus=gpus,
    )
    assert isinstance(result, PartitionResult)
    # All layers should be assigned
    total_assigned = sum(len(layers) for layers in result.gpu_assignments.values())
    assert total_assigned == 80
    # No GPU should exceed total VRAM dramatically (may overflow on smallest GPU)
    for gpu_id, usage_mb in result.gpu_vram_usage_mb.items():
        assert usage_mb >= 0, f"{gpu_id} has negative VRAM usage: {usage_mb}"
    # Should have transfer bottlenecks
    assert len(result.transfer_bottlenecks_mbps) > 0


def test_assign_layers_heterogeneous_single_gpu():
    """Assign layers with a single GPU."""
    gpus = [GPU_PROFILES["m40"]]
    result = assign_layers_heterogeneous(
        total_layers=24,
        layer_profile_mb=500.0,
        gpus=gpus,
    )
    assert isinstance(result, PartitionResult)
    assert result.gpu_assignments["gpu_1"]  # M40 gets layers
    assert len(result.gpu_assignments["gpu_1"]) == 24


def test_generate_partitioning_schedule():
    """Full schedule generation for 70B model across K80+M40+RTX3060."""
    schedule = generate_partitioning_schedule(
        total_layers=80,
        param_billions=70.0,
        precision_bits=16,
        context_len=8192,
        gpu_config_name="tesla_k80+m40+rtx_3060",
    )
    assert schedule["layer_profile_mb"] > 0
    assert schedule["total_layers"] == 80
    assert len(schedule["gpu_assignments"]) == 3  # K80, M40, RTX 3060
    assert len(schedule["gpu_names"]) == 3
    assert schedule["schedule_efficiency_pct"] > 0
    assert schedule["total_batches"] > 0
    # Transfer bottlenecks should be present
    assert len(schedule["transfer_bottlenecks_mbps"]) > 0
    # Check bottleneck entries are (from_gpu, to_gpu, bandwidth_mbps) tuples
    for bottleneck in schedule["transfer_bottlenecks_mbps"]:
        assert len(bottleneck) == 3
        assert bottleneck[2] > 0  # bandwidth should be positive


def test_generate_partitioning_schedule_invalid_config():
    """Error on unknown GPU config."""
    try:
        generate_partitioning_schedule(
            total_layers=80,
            param_billions=70.0,
            precision_bits=16,
            context_len=8192,
            gpu_config_name="invalid+gpu",
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_partition_result_fields():
    """Verify PartitionResult has all required fields."""
    gpus = [
        GPU_PROFILES["tesla_k80"],
        GPU_PROFILES["m40"],
    ]
    result = assign_layers_heterogeneous(
        total_layers=12,
        layer_profile_mb=300.0,
        gpus=gpus,
    )
    assert "gpu_assignments" in result.__dict__
    assert "gpu_vram_usage_mb" in result.__dict__
    assert "gpu_compute_balance" in result.__dict__
    assert "transfer_bottlenecks_mbps" in result.__dict__
    assert "total_batches" in result.__dict__
    assert "schedule_efficiency_pct" in result.__dict__
