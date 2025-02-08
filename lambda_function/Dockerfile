FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN mkdir -p ./model

COPY ./model/*.pth ./model
COPY bird_similarity.py .
COPY 4.jpg .

CMD [ "bird_similarity.handler" ]