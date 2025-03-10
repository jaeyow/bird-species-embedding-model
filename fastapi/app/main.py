from fastapi import FastAPI
from app.api.endpoints.router import router

app = FastAPI()

app.include_router(router)