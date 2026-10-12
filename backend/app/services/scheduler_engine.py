"""Domain logic for layer batching schedule optimization."""
import math
from typing import List
from app.models.schemas import ScheduleRequest, ScheduleResult, ScheduleBatchChunk

class SchedulerEngine:
    @staticmethod
    def calculate_optimal_schedule(req: ScheduleRequest) -> ScheduleResult:
        """
        Calculates how many layers can be batched simultaneously into GPU VRAM
        to minimize disk/CPU-to-GPU PCIe transfer overhead and context switching.
        """
        # Reserve 15% safety buffer for CUDA context and activations
        usable_vram_mb = req.available_vram_mb * 0.85

        # Maximum layers that fit in usable VRAM
        raw_layers_per_batch = math.floor(usable_vram_mb / max(1.0, req.single_layer_mb))
        optimal_layers = max(1, min(req.total_layers, int(raw_layers_per_batch)))

        total_batches = math.ceil(req.total_layers / optimal_layers)
        batch_vram_usage_mb = optimal_layers * req.single_layer_mb
        vram_utilization_pct = min(100.0, (batch_vram_usage_mb / req.available_vram_mb) * 100.0)

        # Transfer model:
        # Transfer time = (Size in MB) / (PCIe bandwidth in MB/s) + Fixed PCIe launch overhead (1.5ms per transfer)
        pcie_mbps = req.pcie_bandwidth_gbps * 1024.0 / 8.0  # GB/s to MB/s
        transfer_overhead_ms = 1.5

        chunks: List[ScheduleBatchChunk] = []
        accumulated_transfer_sec = 0.0

        for b in range(total_batches):
            start_l = b * optimal_layers
            end_l = min(req.total_layers - 1, (b + 1) * optimal_layers - 1)
            count = end_l - start_l + 1
            chunk_vram_mb = count * req.single_layer_mb
            # Data transfer in milliseconds
            transfer_ms = (chunk_vram_mb / pcie_mbps * 1000.0) + transfer_overhead_ms
            accumulated_transfer_sec += (transfer_ms / 1000.0)

            chunks.append(
                ScheduleBatchChunk(
                    chunk_index=b,
                    start_layer=start_l,
                    end_layer=end_l,
                    layers_count=count,
                    vram_usage_mb=round(chunk_vram_mb, 2),
                    transfer_time_ms=round(transfer_ms, 2)
                )
            )

        # Compare with single layer loading time (which incurs total_layers * transfer_overhead_ms + unbatched small transfers)
        # In single layer loading, small chunk transfers have lower effective PCIe throughput (e.g. 50% efficiency for tiny packets) plus individual kernel launch overhead
        single_transfer_ms_each = (req.single_layer_mb / (pcie_mbps * 0.75) * 1000.0) + transfer_overhead_ms
        single_layer_total_sec = (req.total_layers * single_transfer_ms_each) / 1000.0
        speedup = single_layer_total_sec / max(0.001, accumulated_transfer_sec)

        return ScheduleResult(
            optimal_layers_per_batch=optimal_layers,
            total_batches=total_batches,
            vram_utilization_pct=round(vram_utilization_pct, 2),
            total_transfer_time_seconds=round(accumulated_transfer_sec, 3),
            chunks=chunks,
            speedup_vs_single_layer=round(max(1.0, speedup), 2)
        )
