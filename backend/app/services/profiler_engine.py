"""Domain logic for layer-by-layer parameter and memory profiling."""
from typing import List, Tuple
from app.models.schemas import ProfileRequest, ProfileResult, LayerProfileItem

class ProfilerEngine:
    @staticmethod
    def calculate_layer_memory(
        param_billions: float,
        precision_bits: int,
        total_layers: int,
        context_len: int,
        hidden_dim: int = 8192
    ) -> Tuple[float, float, float]:
        """
        Calculates parameter size, activation memory, and KV cache memory per layer.
        Returns: (param_size_mb, activation_mb, kv_cache_mb)
        """
        # Bytes per parameter: e.g. 4 bits = 0.5 bytes, 8 bits = 1.0 byte, 16 bits = 2.0 bytes
        bytes_per_param = precision_bits / 8.0
        total_param_bytes = param_billions * 1e9 * bytes_per_param
        total_param_mb = total_param_bytes / (1024 * 1024)

        # Base layer parameter overhead (approx 90% in transformer layers, 10% in embeddings/heads)
        transformer_params_mb = total_param_mb * 0.90
        per_layer_param_mb = transformer_params_mb / max(1, total_layers)

        # Activation memory per layer during forward pass (batch size = 1)
        # Activation is approx (2 * seq_len * hidden_dim * bytes_per_activation)
        activation_bytes = 2 * context_len * hidden_dim * 2.0  # FP16 activations
        activation_mb = activation_bytes / (1024 * 1024)

        # KV cache memory per layer (2 * 2 * n_heads_kv * head_dim * context_len * bytes)
        # For standard 70B architecture: GQA with 8 KV heads, head_dim 128
        kv_cache_bytes = 2 * 2 * 8 * 128 * context_len * 2.0
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        return per_layer_param_mb, activation_mb, kv_cache_mb

    @classmethod
    def run_profiling(cls, req: ProfileRequest) -> ProfileResult:
        param_mb, act_mb, kv_mb = cls.calculate_layer_memory(
            param_billions=req.parameter_count_billions,
            precision_bits=req.precision_bits,
            total_layers=req.total_layers,
            context_len=req.context_length
        )

        single_layer_vram_mb = param_mb + act_mb + kv_mb
        full_model_vram_gb = (req.parameter_count_billions * (req.precision_bits / 8.0) * 1e9) / (1024 ** 3)

        target_vram_mb = req.target_vram_gb * 1024.0
        # In AirLLM layer pipelining, only single layer (or batch of layers) + base weights are in VRAM
        fits = single_layer_vram_mb <= (target_vram_mb * 0.85)  # 15% safety buffer for CUDA runtime
        vram_headroom_mb = max(0.0, target_vram_mb - single_layer_vram_mb)

        layers: List[LayerProfileItem] = []
        for i in range(req.total_layers):
            layers.append(
                LayerProfileItem(
                    layer_index=i,
                    name=f"model.layers.{i}",
                    param_size_mb=round(param_mb, 2),
                    activation_size_mb=round(act_mb, 2),
                    kv_cache_size_mb=round(kv_mb, 2),
                    cumulative_vram_mb=round(single_layer_vram_mb, 2)
                )
            )

        return ProfileResult(
            model_name=req.model_name,
            parameter_count_billions=req.parameter_count_billions,
            precision_bits=req.precision_bits,
            total_layers=req.total_layers,
            target_vram_gb=req.target_vram_gb,
            single_layer_vram_mb=round(single_layer_vram_mb, 2),
            full_model_vram_gb=round(full_model_vram_gb, 2),
            fits_in_target_vram=fits,
            vram_headroom_mb=round(vram_headroom_mb, 2),
            layers=layers
        )
