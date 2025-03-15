from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.endpoints.router import router
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# More permissive CORS settings for debugging
origins = [
    "*"                        # Allow all origins for testing
]

logger.info(f"Configuring CORS with allowed origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex="http://localhost:.*",  # Allow any localhost port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory for bird images
# The path is relative to where uvicorn is run
app.mount("/birds", StaticFiles(directory="all_birds"), name="birds")

# Include the router
app.include_router(router)

