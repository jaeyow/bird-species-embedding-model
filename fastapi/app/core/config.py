from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FastAPI Azure API"
    admin_email: str
    secret_key: str
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

settings = Settings()