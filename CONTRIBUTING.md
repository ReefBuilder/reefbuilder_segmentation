# Contributing

## Setup

### 1. Install Python

Create and activate a conda virtual environment based on python 3.11
```commandline
conda create -n coral-detection python=3.11
conda activate coral-detection
```

### 2. Install Poetry
This repo uses Poetry to manage Python dependencies. The `poetry.lock` file records the exact package versions.

Poetry installation instructions: https://python-poetry.org/docs/

Run the following to install all required dependencies:

```shell
poetry install
```

When new dependencies are required, add them to the project using `poetry add` and run `poetry lock` to resolve the dependencies (which will update the `poetry.lock` file, which should be committed).

