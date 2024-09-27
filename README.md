# bird-species-embedding-model
Building an image embedding model using timm PyTorch library
# Have now started the move to using mamba package manager
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