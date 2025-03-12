# bird-species-embedding-model
Building an image embedding model using [timm PyTorch library](https://timm.fast.ai/).

## Notebooks

- [Prepare dataset](notebooks/00-prepare-dataset.ipynb): Create a smaller subset of 50 bird species.

- [Using a pre-trained model](notebooks/01-using-pretrained-model.ipynb): Use a pre-trained model to extract features from the dataset.

- [Fine-tuning a model](notebooks/02-fine-tuning-a-pre-trained-model.ipynb): Fine-tune a pre-trained model on the dataset.


## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd src/fastapi
   ```

2. **Create a virtual environment:**
   ```bash
   pyenv versions
   pyenv install 3.12.8
   # Set the local python version
   pyenv local 3.12.8
   python -m venv .venv
   source .venv/bin/activate

   # then you can check the python version
   python --version

   # and where the python is located
   which python
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   # run the API locally in normal mode
   docker compose up --build

   # run the API locally in debug mode (you can add breakpoints and inspect code)
   docker compose -f docker-compose-debug.yml up --build
   ```

## Here are some conda/mamba commands to get you started:

```bash
# install mamba package manager
conda install mamba -n base -c conda-forge

# create a new environment
mamba create -n bird-species-env python==3.10.14

# activate the new environment
mamba activate bird-species-env

# list environments
maamba env list

# remove an environment
mamba env remove -n bird-species-env-2

# initialize the emamba first
mamba init zsh

# source the mamba shell
source ~/.zshrc

# activate the new environment
mamba activate bird-species-env

# create export environment to a file
mamba env export > bird-species-env.yml

# install packages from the environment file
mamba env create -f bird-species-env.yml

# deactivate the environment
mamba deactivate
```
