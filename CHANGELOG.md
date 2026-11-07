# Changelog

## 1.3.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.2.0 — 2026-11-05

### Added
- Combined Profile + Auto-Schedule endpoint (`POST /api/profile-and-schedule`) that runs memory profiling and automatically computes optimal layer batching schedule using the profiler's single-layer VRAM calculation.
- New request/response schemas: `ProfileAndScheduleRequest`, `ProfileAndScheduleResult`.
- Integration test for combined workflow verifying end-to-end profile persistence and schedule optimization.

### Fixed
- Sync version between `pyproject.toml` (1.0.0) and `VERSION` (1.2.0) to 1.2.0.
- Update README version reference to 1.2.0.

## 1.1.0 — work-in-progress checkpoint

### Added
- Export analysis results to JSON/CSV for all workflows:
  - Memory Profiler: exports profile configuration, simulation results, and layer-by-layer breakdown
  - Layer Scheduler: exports schedule configuration, optimization results, and batch chunk details
  - Diagnostic Console: exports diagnostic request parameters, compatibility results, and identified issues
  - Audit Log View: exports all audit trail entries
- Export buttons integrated into each workflow results section
- CSS styling for export actions and header button groups

### Fixed
- Resolved Pydantic v2 protected namespace deprecation warnings for `model_name` fields in `ProfileRequest` and `ProfileResult` schemas by adding `model_config = ConfigDict(protected_namespaces=())`.
- Migrated `Settings` class from deprecated class-based `Config` to `model_config = SettingsConfigDict(...)` for pydantic-settings v2 compatibility.
- Eliminated all deprecation warnings from test runs; 18 backend and 3 frontend tests pass cleanly.

Partial code published for continued development. Known issues: STATUS.md.

## [1.0.1] - 2026-10-21

### Fixed
- Resolved Pydantic v2 protected namespace deprecation warnings for `model_name` fields in `ProfileRequest` and `ProfileResult` schemas by adding `model_config = ConfigDict(protected_namespaces=())`.
- Migrated `Settings` class from deprecated class-based `Config` to `model_config = SettingsConfigDict(...)` for pydantic-settings v2 compatibility.
- Eliminated all deprecation warnings from test runs; 18 backend and 3 frontend tests pass cleanly.

Partial code published for continued development. Known issues: STATUS.md.

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-10-12

### Added
- Layer-pipelined memory estimation and step-by-step layer transfer simulator for 70B/405B LLMs.
- Multi-layer batch scheduling engine solving the optimal VRAM utilization trade-off for 4GB, 16GB, and 24GB GPUs.
- Compatibility diagnostic analyzer resolving rotary embedding, dynamic cache, stateful mixin, and CUDA/ROCm environment failures.
- Fast, deterministic offline inference pipeline simulation and advisory opt-in multi-provider LLM explainers.
- FastAPI backend with SQLite persistence, audit logging, role-based authorization, and OIDC Keycloak support.
- React TypeScript dashboard with interactive memory visualizers, schedule planners, and diagnostic reports.

---

Alan Vo · [alanvo.net](https://alanvo.net) · [X: @alanvoasi](https://x.com/alanvoasi)
