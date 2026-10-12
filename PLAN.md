# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

layer-pipelined-memory-profiler

## Roadmap

### layer-pipelined-memory-profiler — Layer-Pipelined Memory Profiler and Layer Batching Engine

Status: done

Acceptance:
- Simulate layer-by-layer parameter loading and VRAM allocation for 70B models on low-VRAM targets
- Compute optimal batch layer chunk sizes based on available GPU VRAM (1.6GB to 24GB)
- Diagnose runtime issues such as DynamicCache, rotary_emb, and ROCm/CUDA configuration faults
- Deterministic offline core execution with SQLite audit trail and persistent simulations
- Advisory LLM analysis adapter supporting OpenAI-compatible, Anthropic, Gemini, and Ollama
- OIDC auth flow with mock test harness and local demo mode

Implementation: `backend/app/main.py`, `backend/app/core/config.py`, `backend/app/core/security.py`, `backend/app/models/schemas.py`, `backend/app/models/database.py`, `backend/app/services/profiler_engine.py`, `backend/app/services/scheduler_engine.py`, `backend/app/services/diagnostic_engine.py`, `backend/app/services/llm_advisory.py`, `backend/app/api/routes.py`, `frontend/src/App.tsx`, `frontend/src/components/MemoryProfiler.tsx`, `frontend/src/components/LayerScheduler.tsx`, `frontend/src/components/DiagnosticConsole.tsx`, `frontend/src/components/AuditLogView.tsx`, `frontend/src/api/client.ts`

Tests: `backend/tests/test_profiler.py`, `backend/tests/test_scheduler.py`, `backend/tests/test_diagnostics.py`, `backend/tests/test_api.py`, `backend/tests/test_auth.py`, `frontend/src/App.test.tsx`

### multi-gpu-heterogeneous-mesh — Heterogeneous Multi-GPU Pipeline Partitioning

Status: planned

Acceptance:
- Partition transformer layers across asymmetric legacy GPUs (e.g. Tesla K80 + M40 + RTX 3060)
- Compute interconnect PCIe and NVLink bandwidth transfer bottlenecks
- Generate multi-device layer scheduling tables

### real-time-hardware-telemetry — Live NVML & ROCm SMI Hardware Telemetry Stream

Status: planned

Acceptance:
- Stream live VRAM usage, PCIe RX/TX throughput, and GPU compute utilization via WebSocket
- Compare theoretical AirLLM layer transfer speeds against observed real-world disk-to-VRAM transfers

