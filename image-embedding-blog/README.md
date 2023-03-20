# Image Embeddings Blog
Supplementary code for our blog on Pre-Trained Image Embeddings with HuggingFace.

## Pre-requisites
You'll need following pre-requisites to run the demo:

- [Python 3](https://www.python.org/downloads/)

## Getting started

1. Clone this repo:
```bash
git clone git@github.com:fuzzylabs/image-embeddings-blog.git
```

2. Go to the cloned directory:
```bash
cd image-embeddings-blog
```

3. Create a new Python virtual environment and activate it. For Linux/MacOS users:
```bash
python3 -m venv demoenv
source demoenv/bin/activate
pip install -r requirements.txt
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

Run the pipeline from the root directory of this repo:

```bash
$ python3 zenml-pipeline/run.py
```
