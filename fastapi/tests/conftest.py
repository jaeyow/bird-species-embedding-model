import pytest

@pytest.fixture(scope="session")
def test_client():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    yield client