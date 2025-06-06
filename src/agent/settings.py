from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Don't require .env file to exist
        env_ignore_empty=True,
        # Use environment variables from system
        case_sensitive=True,
    )

    # AI API Keys
    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    TOGETHER_API_KEY: str

    # WhatsApp API Configuration
    WHATSAPP_TOKEN: str
    WHATSAPP_PHONE_NUMBER_ID: str
    WHATSAPP_VERIFY_TOKEN: str

    # Database configuration
    DATABASE_URL: str
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # Vector Database Configuration
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_URL: str
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: Optional[str] = None

    # Model Configuration
    TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    SMALL_TEXT_MODEL_NAME: str = "gemma2-9b-it"
    STT_MODEL_NAME: str = "whisper-large-v3-turbo"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"
    ITT_MODEL_NAME: str = "llama-3.2-11b-vision-preview"

    # Application Configuration
    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    # Environment
    TESTING: str = "false"


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings instance, creating it if needed."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# DO NOT instantiate settings at module level - this causes validation errors at import time
# Use get_settings() instead wherever you need settings
