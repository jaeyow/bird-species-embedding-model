.PHONY: run

# cli: make -f utility.mak run
run:
	docker compose -f docker-compose.yml up --build

# deploy:
# 	# Replace with your ECR repository URI
# 	docker tag bird-similarity-api:latest <your-ecr-repo-uri>:latest
# 	docker push <your-ecr-repo-uri>:latest