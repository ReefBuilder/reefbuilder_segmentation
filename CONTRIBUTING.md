# Contributing

## Setup

### 1. Install Poetry using pipx
This repo uses Poetry to manage Python dependencies. The `poetry.lock` file records the exact package versions.

Poetry installation instructions: https://python-poetry.org/docs/

Install pipx first and then install poetry using it.

### 2. Create conda environment

Create and activate a conda virtual environment based on python 3.11. If conda isn't already installed, then please 
follow [this link](https://docs.anaconda.com/miniconda/#quick-command-line-install).
```commandline
conda create -n reefbuilder_segmentation python=3.11
conda activate reefbuilder_segmentation
```

### 3. Install Packages

Run the following to install all required dependencies:

```shell
poetry install --no-root
```

When new dependencies are required, add them to the project using `poetry add` and run `poetry lock` to resolve
the dependencies (which will update the `poetry.lock` file, which should be committed).


### 4. Create Git commit message template
This repo uses a specific format for git commits. This format can be found in the ".gitmessage" file.

Run the following to automatically initialise the template in the default commit message:

```shell
 git config --local commit.template .gitmessage
```

### 5. Format code using Black
Black is being used as the formatter. Post the above steps it should already be installed in your
environment. Either follow online instructions to set up Black with your IDE of choice, or you can 
simply run Black using the terminal. Documentation can be found [here](https://black.readthedocs.io/en/stable/).
