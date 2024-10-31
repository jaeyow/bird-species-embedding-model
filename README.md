# bird-species-embedding-model
Building an image embedding model using [timm PyTorch library](https://timm.fast.ai/).

## Notebooks

- [Prepare dataset](notebooks/00-prepare-dataset.ipynb): Create a smaller subset of 50 bird species.

- [Using a pre-trained model](notebooks/01-using-pretrained-model.ipynb): Use a pre-trained model to extract features from the dataset.

- [Fine-tuning a model](notebooks/02-fine-tuning-a-pre-trained-model.ipynb): Fine-tune a pre-trained model on the dataset.

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
