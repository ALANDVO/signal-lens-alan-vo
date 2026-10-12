"""API endpoints for SignalLens workflows."""
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models.database import (
    get_db,
    ProfileRecord,
    ScheduleRecord,
    DiagnosticRecord,
    AuditLogRecord,
    utcnow
)
from app.models.schemas import (
    ProfileRequest,
    ProfileResult,
    LayerProfileItem,
    ScheduleRequest,
    ScheduleResult,
    DiagnosticRequest,
    DiagnosticResult,
    AdvisoryRequest,
    AdvisoryResponse,
    AuditLogItem
)
from app.core.security import get_current_user, require_role, AuthenticatedUser
from app.services.profiler_engine import ProfilerEngine
from app.services.scheduler_engine import SchedulerEngine
from app.services.diagnostic_engine import DiagnosticEngine
from app.services.llm_advisory import LLMAdvisoryService

router = APIRouter()

async def log_action(db: AsyncSession, action: str, user_sub: str, details: str):
    audit = AuditLogRecord(action=action, user_sub=user_sub, details=details, timestamp=utcnow())
    db.add(audit)
    await db.commit()

# --- Workflow 1: Layer-Pipelined Memory Profiling ---
@router.post("/profiles", response_model=ProfileResult)
async def create_profile(
    req: ProfileRequest,
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("operator"))
):
    result = ProfilerEngine.run_profiling(req)
    record = ProfileRecord(
        model_name=result.model_name,
        parameter_count_billions=result.parameter_count_billions,
        precision_bits=result.precision_bits,
        total_layers=result.total_layers,
        target_vram_gb=result.target_vram_gb,
        single_layer_vram_mb=result.single_layer_vram_mb,
        full_model_vram_gb=result.full_model_vram_gb,
        fits_in_target_vram=result.fits_in_target_vram,
        vram_headroom_mb=result.vram_headroom_mb,
        layers_json=json.dumps([l.model_dump() for l in result.layers])
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    result.id = record.id
    result.created_at = record.created_at

    await log_action(db, "CREATE_PROFILE", user.sub, f"Profile created for {req.model_name} (ID: {record.id})")
    return result

@router.get("/profiles", response_model=List[ProfileResult])
async def list_profiles(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("viewer"))
):
    stmt = select(ProfileRecord).order_by(desc(ProfileRecord.id)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    records = res.scalars().all()
    results = []
    for r in records:
        layers = [LayerProfileItem(**item) for item in json.loads(r.layers_json)]
        results.append(
            ProfileResult(
                id=r.id,
                model_name=r.model_name,
                parameter_count_billions=r.parameter_count_billions,
                precision_bits=r.precision_bits,
                total_layers=r.total_layers,
                target_vram_gb=r.target_vram_gb,
                single_layer_vram_mb=r.single_layer_vram_mb,
                full_model_vram_gb=r.full_model_vram_gb,
                fits_in_target_vram=r.fits_in_target_vram,
                vram_headroom_mb=r.vram_headroom_mb,
                layers=layers,
                created_at=r.created_at
            )
        )
    return results

@router.get("/profiles/{profile_id}", response_model=ProfileResult)
async def get_profile(
    profile_id: int,
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("viewer"))
):
    stmt = select(ProfileRecord).where(ProfileRecord.id == profile_id)
    res = await db.execute(stmt)
    r = res.scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Profile not found")
    layers = [LayerProfileItem(**item) for item in json.loads(r.layers_json)]
    return ProfileResult(
        id=r.id,
        model_name=r.model_name,
        parameter_count_billions=r.parameter_count_billions,
        precision_bits=r.precision_bits,
        total_layers=r.total_layers,
        target_vram_gb=r.target_vram_gb,
        single_layer_vram_mb=r.single_layer_vram_mb,
        full_model_vram_gb=r.full_model_vram_gb,
        fits_in_target_vram=r.fits_in_target_vram,
        vram_headroom_mb=r.vram_headroom_mb,
        layers=layers,
        created_at=r.created_at
    )

# --- Workflow 2: Layer Batching Schedule Optimization ---
@router.post("/schedules", response_model=ScheduleResult)
async def create_schedule(
    req: ScheduleRequest,
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("operator"))
):
    result = SchedulerEngine.calculate_optimal_schedule(req)
    record = ScheduleRecord(
        optimal_layers_per_batch=result.optimal_layers_per_batch,
        total_batches=result.total_batches,
        vram_utilization_pct=result.vram_utilization_pct,
        total_transfer_time_seconds=result.total_transfer_time_seconds,
        speedup_vs_single_layer=result.speedup_vs_single_layer,
        chunks_json=json.dumps([c.model_dump() for c in result.chunks])
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    result.id = record.id
    result.created_at = record.created_at

    await log_action(db, "CREATE_SCHEDULE", user.sub, f"Created schedule with batch size {result.optimal_layers_per_batch}")
    return result

@router.get("/schedules", response_model=List[ScheduleResult])
async def list_schedules(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("viewer"))
):
    stmt = select(ScheduleRecord).order_by(desc(ScheduleRecord.id)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    records = res.scalars().all()
    results = []
    for r in records:
        chunks = json.loads(r.chunks_json)
        results.append(
            ScheduleResult(
                id=r.id,
                optimal_layers_per_batch=r.optimal_layers_per_batch,
                total_batches=r.total_batches,
                vram_utilization_pct=r.vram_utilization_pct,
                total_transfer_time_seconds=r.total_transfer_time_seconds,
                speedup_vs_single_layer=r.speedup_vs_single_layer,
                chunks=chunks,
                created_at=r.created_at
            )
        )
    return results

# --- Workflow 3: Diagnostic & Environment Compatibility ---
@router.post("/diagnostics", response_model=DiagnosticResult)
async def run_diagnostic(
    req: DiagnosticRequest,
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("operator"))
):
    result = DiagnosticEngine.analyze_environment(req)
    record = DiagnosticRecord(
        transformers_version=req.transformers_version,
        torch_version=req.torch_version,
        device_type=req.device_type,
        is_compatible=result.is_compatible,
        compatibility_score=result.compatibility_score,
        issues_json=json.dumps([i.model_dump() for i in result.issues_found])
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    result.id = record.id
    result.created_at = record.created_at

    await log_action(db, "RUN_DIAGNOSTIC", user.sub, f"Diagnostic run: score={result.compatibility_score}")
    return result

@router.get("/diagnostics", response_model=List[DiagnosticResult])
async def list_diagnostics(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("viewer"))
):
    stmt = select(DiagnosticRecord).order_by(desc(DiagnosticRecord.id)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    records = res.scalars().all()
    results = []
    for r in records:
        issues = json.loads(r.issues_json)
        results.append(
            DiagnosticResult(
                id=r.id,
                issues_found=issues,
                is_compatible=r.is_compatible,
                compatibility_score=r.compatibility_score,
                created_at=r.created_at
            )
        )
    return results

# --- Advisory Endpoint ---
@router.post("/advisory", response_model=AdvisoryResponse)
async def get_advisory(
    req: AdvisoryRequest,
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("viewer"))
):
    res = await LLMAdvisoryService.get_advisory(req)
    await log_action(db, "REQUEST_ADVISORY", user.sub, f"Advisory for {req.context_type}")
    return res

# --- Audit Logs ---
@router.get("/audit-logs", response_model=List[AuditLogItem])
async def get_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: AuthenticatedUser = Depends(require_role("admin"))
):
    stmt = select(AuditLogRecord).order_by(desc(AuditLogRecord.id)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    records = res.scalars().all()
    return [
        AuditLogItem(
            id=r.id,
            timestamp=r.timestamp,
            action=r.action,
            user_sub=r.user_sub,
            details=r.details
        )
        for r in records
    ]
