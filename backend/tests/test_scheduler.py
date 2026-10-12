import pytest
from app.services.scheduler_engine import SchedulerEngine
from app.models.schemas import ScheduleRequest

def test_optimal_schedule_batching_4gb():
    # 4GB GPU (~3800MB usable after OS display buffer)
    req = ScheduleRequest(
        total_layers=80,
        single_layer_mb=450.0,
        available_vram_mb=3800.0,
        pcie_bandwidth_gbps=12.0
    )
    res = SchedulerEngine.calculate_optimal_schedule(req)
    # usable vram is 3800 * 0.85 = 3230 MB -> floor(3230 / 450) = 7 layers
    assert res.optimal_layers_per_batch == 7
    assert res.total_batches == 12  # ceil(80 / 7)
    assert res.vram_utilization_pct > 70.0
    assert res.speedup_vs_single_layer > 1.0
    assert len(res.chunks) == 12

def test_optimal_schedule_single_layer_on_tiny_vram():
    req = ScheduleRequest(
        total_layers=40,
        single_layer_mb=800.0,
        available_vram_mb=900.0,  # very tight VRAM
        pcie_bandwidth_gbps=8.0
    )
    res = SchedulerEngine.calculate_optimal_schedule(req)
    assert res.optimal_layers_per_batch == 1
    assert res.total_batches == 40

def test_chunk_boundaries_consistent():
    req = ScheduleRequest(
        total_layers=35,
        single_layer_mb=300.0,
        available_vram_mb=4000.0,
        pcie_bandwidth_gbps=16.0
    )
    res = SchedulerEngine.calculate_optimal_schedule(req)
    assert res.chunks[0].start_layer == 0
    assert res.chunks[-1].end_layer == 34
