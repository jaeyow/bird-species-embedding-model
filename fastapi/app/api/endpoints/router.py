import json
from typing import List
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import StartJobRequest, StartJobResponse, JobStatusResponse, JobResultResponse
from app.services.service import LlmJob, LlmJobService
router = APIRouter()

@router.post(
    "/jobs",
    tags=["Jobs"],
    summary="Start a new question generation job",
    response_model=StartJobResponse)
async def start_llm_job(request: StartJobRequest):
    service = LlmJobService()
    llm_job: LlmJob = await service.create_job(request)
    print(f"Created job {llm_job.job_id} with parameters {request}")
    return StartJobResponse(job_id=llm_job.job_id, status=llm_job.status)

@router.get(
    "/jobs/{job_id}/status",
    tags=["Job status"],
    summary="Get the status of a question generation job by job ID",
    response_model=JobStatusResponse)
async def llm_job_status(job_id: str):
    service = LlmJobService()
    job = await service.get_job(job_id)
    print(f"Job: {job_id} status: {job.status}")
    if job.status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job_id, status=job.status)

@router.get(
    "/jobs/{job_id}/results",
    tags=["Job results"],
    summary="Get the results of a question generation job by job ID",
    response_model=JobResultResponse)
async def llm_job_result(job_id: str):
    print(f"Getting result for job {job_id}")
    service = LlmJobService()
    result_data = {"result": "Blah"}
    result = await service.complete_job(job_id, result_data)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found or result not available")
    return JobResultResponse(job_id=job_id, result=result)
