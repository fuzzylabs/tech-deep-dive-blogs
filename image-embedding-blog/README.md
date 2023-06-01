# Image Embeddings Blog
Supplementary code for our blog on Pre-Trained Image Embeddings with HuggingFace.

## Pre-requisites
You'll need following pre-requisites to run the demo:

- [Python 3](https://www.python.org/downloads/)
- [Conda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Getting started

1. Clone this repo:
```bash
git clone git@github.com:fuzzylabs/tech-deep-dive-blogs.git
```

2. Go to the cloned directory:
```bash
cd image-embeddings-blog
```

3. Create a new Python virtual environment with Conda, activate it and install required libraries. For Linux/MacOS users:
```bash
conda create -y --name demoenv python==3.9.0
conda activate demoenv
pip install -r requirements.txt
```

When you are finished with this tutorial you exit the Conda environment using:
```bash
conda deactivate
```

The `data` directory contains all images required for this tutorial.

Use the `embbeddings.ipynb` notebook to follow along with the blog.

# ZenML Pipeline

The code within `zenml-pipeline` can be run using the following steps:

Initialise ZenML:

```bash
$ zenml init
```

Create ZenML server:

```bash
$ zenml up
```

Run the pipeline from the `image-embedding-blog` directory:

```bash
$ python3 zenml-pipeline/run.py
```
