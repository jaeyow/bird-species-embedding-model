# Use the official AWS Lambda base image for Python 3.10
FROM public.ecr.aws/lambda/python:3.1

RUN pip install timm torch torchvision

COPY bird_similarity.py ${LAMBDA_TASK_ROOT}

CMD ["bird_similarity.handler"]
