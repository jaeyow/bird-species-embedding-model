from typing import List, Dict, Any
from fastapi import HTTPException
import uuid
import json
import os

from app.schemas.schemas import StartJobRequest

class LlmJob:
    def __init__(self, job_id: str, status: str, result: Any = None):
        self.job_id = job_id
        self.status = status
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LlmJob':
        return cls(
            job_id=data["job_id"],
            status=data["status"],
            result=data.get("result")
        )

class LlmJobService:
    def __init__(self, file_path: str = "jobs.json"):
        self.jobs: Dict[str, LlmJob] = {}
        self.file_path = file_path
        self.load_jobs()
        print(f"Loaded {len(self.jobs)} jobs")

    async def create_job(self, request: StartJobRequest) -> LlmJob:
        job_id = str(uuid.uuid4())
        print(f"Creating job {job_id} with prop1 {request.prop1}")
        job = LlmJob(job_id=job_id, status="created")
        self.jobs[job_id] = job
        self.save_jobs()
        return job

    async def get_job(self, job_id: str) -> LlmJob:
        print(f"Getting job status for {job_id}")
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    async def complete_job(self, job_id: str, result: Any) -> LlmJob:
        job = await self.get_job(job_id)
        job.status = "completed"
        job.result = result
        self.save_jobs()
        return job.result

    async def list_jobs(self, status: str = None) -> List[LlmJob]:
        print(f"len(self.jobs): {len(self.jobs)}")
        if status:
            return [job for job in self.jobs.values() if job.status == status]
        return list(self.jobs.values())

    def save_jobs(self):
        new_jobs = [job.to_dict() for job in self.jobs.values()]

        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                existing_jobs = json.load(f)
                existing_jobs_dict = {job["job_id"]: job for job in existing_jobs}
        else:
            existing_jobs_dict = {}

        # Update existing jobs with new jobs
        for job in new_jobs:
            existing_jobs_dict[job["job_id"]] = job

        with open(self.file_path, "w") as f:
            json.dump(list(existing_jobs_dict.values()), f)

    def load_jobs(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                jobs_data = json.load(f)
                self.jobs = {job["job_id"]: LlmJob.from_dict(job) for job in jobs_data}