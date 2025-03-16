FROM python:3.12.8

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app ./app

# Create directory for bird images
COPY ./all_birds ./app/all_birds

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--log-level", "debug"]