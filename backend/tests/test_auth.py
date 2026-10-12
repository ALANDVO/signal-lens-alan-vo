import pytest
from httpx import AsyncClient, ASGITransport
from jose import jwt
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.main import app
from app.models.database import Base, get_db
from app.core.config import settings

import pytest_asyncio
from sqlalchemy.pool import StaticPool

@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_auth_rejection_when_oidc_enabled():
    # Temporarily set OIDC_ENABLED = True and APP_ENV = 'production'
    settings.OIDC_ENABLED = True
    settings.APP_ENV = "production"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request without token must be rejected with 401
        resp = await client.get("/api/profiles")
        assert resp.status_code == 401
        assert "Missing Bearer" in resp.json()["detail"]

    # Reset back to demo
    settings.OIDC_ENABLED = False
    settings.APP_ENV = "demo"

@pytest.mark.asyncio
async def test_role_authorization_viewer_denied_operator():
    settings.OIDC_ENABLED = True
    settings.APP_ENV = "production"

    # Create a viewer token with only "viewer" role
    token = jwt.encode(
        {"sub": "viewer_user", "preferred_username": "viewer", "roles": ["viewer"]},
        "mock_secret_key",
        algorithm="HS256"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Viewer tries to perform operator-level mutation (POST /api/profiles)
        resp = await client.post(
            "/api/profiles",
            json={
                "model_name": "test-model",
                "parameter_count_billions": 7.0,
                "precision_bits": 4,
                "total_layers": 32,
                "context_length": 1024,
                "target_vram_gb": 4.0
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 403
        assert "operator" in resp.json()["detail"]

    settings.OIDC_ENABLED = False
    settings.APP_ENV = "demo"

@pytest.mark.asyncio
async def test_operator_allowed_mutation():
    settings.OIDC_ENABLED = True
    settings.APP_ENV = "production"

    # Create an operator token with "operator" role
    token = jwt.encode(
        {"sub": "op_user", "preferred_username": "operator", "roles": ["operator"]},
        "mock_secret_key",
        algorithm="HS256"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/profiles",
            json={
                "model_name": "test-model-op",
                "parameter_count_billions": 7.0,
                "precision_bits": 4,
                "total_layers": 32,
                "context_length": 1024,
                "target_vram_gb": 4.0
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.json()["model_name"] == "test-model-op"

    settings.OIDC_ENABLED = False
    settings.APP_ENV = "demo"
