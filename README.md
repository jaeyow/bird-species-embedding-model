# bird-species-embedding-model
Building an image embedding model using timm PyTorch library

```bash
pyenv versions
pyenv install 3.12.2
pyenv local 3.12.2
python --version # to confirm the version
eval "$(pyenv init -)" # if the version is not 3.12.2
python -m venv .venv
source .venv/bin/activate


pip freeze > requirements.txt
pip install -r requirements.txt -t ./lambda_layer/
``` 