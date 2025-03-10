from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_llm_job():
    response = client.post("/llm/jobs", json={"input": "test input"})
    assert response.status_code == 201
    assert "job_id" in response.json()

def test_get_llm_job_status():
    response = client.get("/llm/jobs/1/status")
    assert response.status_code == 200
    assert "status" in response.json()

def test_get_llm_job_result():
    response = client.get("/llm/jobs/1/result")
    assert response.status_code == 200
    assert "result" in response.json()