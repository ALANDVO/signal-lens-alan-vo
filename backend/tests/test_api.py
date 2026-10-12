import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.main import app
from app.models.database import Base, get_db
from app.core.config import settings

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture
async def async_client():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    resp = await async_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["app"] == "SignalLens"

@pytest.mark.asyncio
async def test_create_and_get_profile_workflow(async_client: AsyncClient):
    payload = {
        "model_name": "Llama-2-70b-chat",
        "parameter_count_billions": 70.0,
        "precision_bits": 4,
        "total_layers": 80,
        "context_length": 2048,
        "target_vram_gb": 4.0
    }
    resp = await async_client.post("/api/profiles", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    profile_id = data["id"]
    assert profile_id is not None
    assert data["fits_in_target_vram"] is True

    # Fetch profile by id
    get_resp = await async_client.get(f"/api/profiles/{profile_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["model_name"] == "Llama-2-70b-chat"

    # List profiles
    list_resp = await async_client.get("/api/profiles")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

@pytest.mark.asyncio
async def test_create_and_list_schedules(async_client: AsyncClient):
    payload = {
        "total_layers": 80,
        "single_layer_mb": 450.0,
        "available_vram_mb": 3800.0,
        "pcie_bandwidth_gbps": 12.0
    }
    resp = await async_client.post("/api/schedules", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimal_layers_per_batch"] == 7

    list_resp = await async_client.get("/api/schedules")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

@pytest.mark.asyncio
async def test_run_diagnostic_workflow(async_client: AsyncClient):
    payload = {
        "transformers_version": "4.41.0",
        "torch_version": "2.3.0",
        "has_cuda": True,
        "device_type": "cuda"
    }
    resp = await async_client.post("/api/diagnostics", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_compatible"] is False
    assert len(data["issues_found"]) >= 1

@pytest.mark.asyncio
async def test_advisory_deterministic_fallback(async_client: AsyncClient):
    payload = {
        "context_type": "profile",
        "context_data": {
            "model_name": "Qwen-7B",
            "fits_in_target_vram": True,
            "single_layer_vram_mb": 220.0,
            "full_model_vram_gb": 3.8
        }
    }
    resp = await async_client.post("/api/advisory", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Deterministic Analysis" in data["advisory_text"]
    assert data["is_simulated"] is True
