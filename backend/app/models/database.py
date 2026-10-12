"""Database models and SQLite connection setup using SQLAlchemy."""
from datetime import datetime, timezone
from typing import AsyncGenerator
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, DateTime, Float, Boolean
from app.core.config import settings

Base = declarative_base()

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class ProfileRecord(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    parameter_count_billions: Mapped[float] = mapped_column(Float)
    precision_bits: Mapped[int] = mapped_column(Integer)
    total_layers: Mapped[int] = mapped_column(Integer)
    target_vram_gb: Mapped[float] = mapped_column(Float)
    single_layer_vram_mb: Mapped[float] = mapped_column(Float)
    full_model_vram_gb: Mapped[float] = mapped_column(Float)
    fits_in_target_vram: Mapped[bool] = mapped_column(Boolean)
    vram_headroom_mb: Mapped[float] = mapped_column(Float)
    layers_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class ScheduleRecord(Base):
    __tablename__ = "schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    optimal_layers_per_batch: Mapped[int] = mapped_column(Integer)
    total_batches: Mapped[int] = mapped_column(Integer)
    vram_utilization_pct: Mapped[float] = mapped_column(Float)
    total_transfer_time_seconds: Mapped[float] = mapped_column(Float)
    speedup_vs_single_layer: Mapped[float] = mapped_column(Float)
    chunks_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class DiagnosticRecord(Base):
    __tablename__ = "diagnostics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transformers_version: Mapped[str] = mapped_column(String(64))
    torch_version: Mapped[str] = mapped_column(String(64))
    device_type: Mapped[str] = mapped_column(String(32))
    is_compatible: Mapped[bool] = mapped_column(Boolean)
    compatibility_score: Mapped[float] = mapped_column(Float)
    issues_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class AuditLogRecord(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    user_sub: Mapped[str] = mapped_column(String(128))
    details: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
