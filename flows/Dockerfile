# Use the official MLflow Docker image as the base image
FROM  ghcr.io/mlflow/mlflow:v2.0.1 

# Set the working directory inside the container
WORKDIR /mlflow

# Expose the port for MLflow
EXPOSE 5000

# Optional: Install additional Python dependencies if needed
RUN pip install boto3

# Command to run the MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--serve-artifacts", "--artifacts-destination", "s3://bird-spcies-embedding-model", "--default-artifact-root", "s3://bird-spcies-embedding-model/artifacts"]
