# Changelog

## 1.1.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.0.0 — work-in-progress checkpoint

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
