# FastAPI Azure API

This project is a FastAPI application designed to manage LLM (Large Language Model) jobs. It provides endpoints for creating, monitoring, and retrieving job statuses and results. The application is structured to facilitate easy deployment on Azure.

## Project Structure

```
fastapi-azure-api
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── endpoints
│   │   │   ├── __init__.py
│   │   │   └── router.py
│   │   └── dependencies.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── models
│   │   ├── __init__.py
│   │   └── models.py
│   ├── schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services
│       ├── __init__.py
│       └── service.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   └── test_api.py
├── .env
├── .gitignore
├── requirements.txt
├── Dockerfile
├── azure-pipelines.yml
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd fastapi-azure-api
   ```

2. **Create a virtual environment:**
   ```bash
   
   pyenv versions
   pyenv install 3.12.8
   pyenv local 3.12.8
   python -m venv .venv
   source .venv/bin/activate
   ```


3. **Install dependencies:**
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   docker compose up --build
   ```

## API Endpoints

The API provides the following endpoints:

- **POST /jobs**: Create a new LLM job.
- **GET /jobs/{job_id}**: Retrieve the status of a specific job.
- **GET /jobs/{job_id}/result**: Retrieve the result of a completed job.

## Testing

To run the tests, use the following command:

```
pytest
```

## Deployment

This application is designed for deployment on Azure. Ensure you have the Azure CLI installed and configured. You can use the provided `azure-pipelines.yml` for CI/CD integration.

## License

This project is licensed under the MIT License. See the LICENSE file for details.