from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.models import LlmJob
from app.schemas.schemas import StartJobRequest

def validate_job_request(job_request: StartJobRequest, db: Session = Depends(get_db)):
    if not job_request.model_name:
        raise HTTPException(status_code=400, detail="Model name must be provided.")
    if not job_request.parameters:
        raise HTTPException(status_code=400, detail="Parameters must be provided.")
    
    existing_job = db.query(LlmJob).filter(LlmJob.model_name == job_request.model_name).first()
    if existing_job:
        raise HTTPException(status_code=400, detail="A job with this model name already exists.")
    
    return job_request

def get_current_user(token: str = Depends(oauth2_scheme)):
    # Logic to retrieve the current user based on the token
    pass