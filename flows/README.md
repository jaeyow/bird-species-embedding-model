
## How to run Metaflow locally

- This folder contains a Dockerfile that builds a custom image of MLflow with the following changes:
  - The default port is changed to 5000
  - The default directory for storing the MLflow runs is changed to /mlflow/mlruns

```bash 
# docker build -t custom-mlflow . # For x86_64
docker build --platform=linux/arm64 -t custom-m2-mlflow .
```

- To run the container, use the following command:
```bash
# docker run -p 5000:5000 -v $(pwd)/mlruns:/mlflow/mlruns custom-mlflow # For x86_64
docker run --platform linux/arm64 -p 5000:5000 -v $(pwd)/mlruns:/mlflow/mlruns custom-m2-mlflow # For Apple M2
```

- To access the MLflow UI, open a browser and go to http://localhost:5000

- The tracking server will then be available at http://localhost:5000

## Metaflow 

```bash
python flow.py run
python flow.py show

python 01-resnet-fine-tuning-flow.py run --epochs 3
```

docker compose --env-file .env up -d --build
docker compose down