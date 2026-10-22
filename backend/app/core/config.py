from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PROJECT_NAME: str = "Video Pipeline Lab - Phase 0"
    DEBUG: bool = True
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    
    # Directorios de Storage
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOADS_DIR: Path = BASE_DIR / "storage" / "uploads"
    ARTIFACTS_DIR: Path = BASE_DIR / "storage" / "artifacts"
    TEMP_DIR: Path = BASE_DIR / "storage" / "temp"
    
    MAX_UPLOAD_SIZE_MB: int = 500
    
    # Capa 3: AI Semántica (Groq prioritario / TokenHarbor / DeepSeek)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    TOKENHARBOR_API_KEY: str = ""
    TOKENHARBOR_BASE_URL: str = "https://tokenharbor.ai/v1"
    LLM_MODEL_NAME: str = "deepseek-v4.1-flash:free"
    
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Asegurar que existan los directorios clave
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
