import pytest
from app.services.profiler_engine import ProfilerEngine
from app.models.schemas import ProfileRequest

def test_single_layer_memory_calculation():
    # 70B model with 4-bit quantization, 80 layers, 2048 context length
    param_mb, act_mb, kv_mb = ProfilerEngine.calculate_layer_memory(
        param_billions=70.0,
        precision_bits=4,
        total_layers=80,
        context_len=2048,
        hidden_dim=8192
    )
    # Total param size at 4-bit is ~35 GB, per layer ~400 MB
    assert 300.0 < param_mb < 500.0
    assert act_mb > 0.0
    assert kv_mb > 0.0

def test_profiler_fits_in_4gb_vram():
    req = ProfileRequest(
        model_name="Meta-Llama-3-70B-bnb-4bit",
        parameter_count_billions=70.0,
        precision_bits=4,
        total_layers=80,
        context_length=2048,
        target_vram_gb=4.0
    )
    res = ProfilerEngine.run_profiling(req)
    assert res.fits_in_target_vram is True
    assert res.single_layer_vram_mb < 2000.0  # single layer uses < 2GB
    assert res.vram_headroom_mb > 1000.0
    assert len(res.layers) == 80

def test_profiler_overflow_condition():
    # 405B model in FP16 with 128k context on 4GB VRAM
    req = ProfileRequest(
        model_name="Meta-Llama-3.1-405B-FP16",
        parameter_count_billions=405.0,
        precision_bits=16,
        total_layers=126,
        context_length=65536,
        target_vram_gb=4.0
    )
    res = ProfilerEngine.run_profiling(req)
    assert res.fits_in_target_vram is False
