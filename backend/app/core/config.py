"""Core application configuration and settings."""
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "SignalLens"
    APP_ENV: str = "demo"  # 'demo' or 'production'
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite+aiosqlite:///./signallens.db"

    # OIDC Configuration
    OIDC_ENABLED: bool = False
    OIDC_ISSUER_URL: str = "http://127.0.0.1:8080/realms/signallens"
    OIDC_CLIENT_ID: str = "signallens-app"
    OIDC_CLIENT_SECRET: Optional[str] = None
    OIDC_AUDIENCE: str = "signallens-app"

    # LLM Settings (Keys only server-side, opt-in)
    LLM_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "openai-compatible"  # openai-compatible, anthropic, gemini, ollama
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_BASE_URL: str = "https://api.openai.com/v1"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
