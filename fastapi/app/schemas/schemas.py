from pydantic import BaseModel
from typing import Optional, List

class StartJobRequest(BaseModel):
    prop1: str
    prop2: Optional[str]
    prop3: Optional[int]
    prop4: Optional[List[str]]
    
class StartJobResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str

class JobResultResponse(BaseModel):
    job_id: str
    result: Optional[dict]
    error: Optional[str]