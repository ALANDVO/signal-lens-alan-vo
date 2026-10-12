"""Pydantic schemas for request validation and response models."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class ProfileRequest(BaseModel):
    model_name: str = Field(..., description="Target model name, e.g., Llama-2-70b, Llama-3-70b, Qwen-7B")
    parameter_count_billions: float = Field(70.0, ge=0.5, le=500.0)
    precision_bits: int = Field(4, description="Quantization bitwidth: 4, 8, 16")
    total_layers: int = Field(80, ge=1, le=200)
    context_length: int = Field(2048, ge=128, le=131072)
    target_vram_gb: float = Field(4.0, ge=1.0, le=96.0)

class LayerProfileItem(BaseModel):
    layer_index: int
    name: str
    param_size_mb: float
    activation_size_mb: float
    kv_cache_size_mb: float
    cumulative_vram_mb: float

class ProfileResult(BaseModel):
    id: Optional[int] = None
    model_name: str
    parameter_count_billions: float
    precision_bits: int
    total_layers: int
    target_vram_gb: float
    single_layer_vram_mb: float
    full_model_vram_gb: float
    fits_in_target_vram: bool
    vram_headroom_mb: float
    layers: List[LayerProfileItem]
    created_at: Optional[datetime] = None

class ScheduleRequest(BaseModel):
    total_layers: int = Field(80, ge=1, le=200)
    single_layer_mb: float = Field(450.0, ge=10.0, le=10000.0)
    available_vram_mb: float = Field(3800.0, ge=500.0, le=100000.0)
    pcie_bandwidth_gbps: float = Field(12.0, ge=1.0, le=64.0)

class ScheduleBatchChunk(BaseModel):
    chunk_index: int
    start_layer: int
    end_layer: int
    layers_count: int
    vram_usage_mb: float
    transfer_time_ms: float

class ScheduleResult(BaseModel):
    id: Optional[int] = None
    optimal_layers_per_batch: int
    total_batches: int
    vram_utilization_pct: float
    total_transfer_time_seconds: float
    chunks: List[ScheduleBatchChunk]
    speedup_vs_single_layer: float
    created_at: Optional[datetime] = None

class DiagnosticRequest(BaseModel):
    transformers_version: str = Field("4.42.0", description="Transformers library version")
    torch_version: str = Field("2.3.0", description="PyTorch version")
    has_cuda: bool = Field(True)
    device_type: str = Field("cuda", description="cuda, rocm, cpu, or directml")
    vram_gb: float = Field(4.0)
    error_log: Optional[str] = Field(None, description="Optional raw error string to parse")

class DiagnosticIssue(BaseModel):
    category: str
    severity: str  # error, warning, info
    summary: str
    root_cause: str
    recommended_fix: str

class DiagnosticResult(BaseModel):
    id: Optional[int] = None
    issues_found: List[DiagnosticIssue]
    is_compatible: bool
    compatibility_score: float
    created_at: Optional[datetime] = None

class AdvisoryRequest(BaseModel):
    context_type: str = Field(..., description="'profile', 'schedule', or 'diagnostic'")
    context_data: Dict[str, Any]
    user_prompt: Optional[str] = None

class AdvisoryResponse(BaseModel):
    advisory_text: str
    provider: str
    model: str
    is_simulated: bool
    grounded_record_id: Optional[str] = None

class AuditLogItem(BaseModel):
    id: int
    timestamp: datetime
    action: str
    user_sub: str
    details: str
