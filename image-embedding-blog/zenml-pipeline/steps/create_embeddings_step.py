"""A step which creates the embeddings for a dataset of preprocessed images."""
from zenml.logger import get_logger
from zenml.steps import step
from datasets import load_dataset, Dataset
from torchvision.datasets.folder import ImageFolder
import torch
from transformers import ViTMAEModel
import numpy as np


logger = get_logger(__name__)


def create_embedding(model, preprocessed_image: torch.Tensor) -> torch.Tensor:
    """Passes a preprocessed image through a pretrained embedding model.

    Args:
        model (PreTrainedModel): Pretrained HuggingFace PyTorch embedding model.
        preprocessed_image (torch.Tensor): Preprocessed image as a PyTorch Tensor

    Returns:
        torch.Tensor: Embedding vector shape (1, 768) as a Tensor
    """
    embedding = model(**preprocessed_image).last_hidden_state[:, 0]

    return np.squeeze(embedding)


@step
def create_embeddings(image_dataset: Dataset) -> Dataset:
    """Create the embeddings for a dataset of preprocessed images.

    Args:
        Dataset: A HuggingFace dataset containing a row of preprocessed images as Tensors.
    
    Returns:
        Dataset: A HuggingFace dataset containing a new column with embeddings as Tensors.
    """
    # Get pretrained model
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    
    image_dataset = image_dataset.map(lambda img: {"embedding": create_embedding(model, img["preprocessed_image"])})

    logger.info(image_dataset['embedding'][0].shape)
    
    return image_dataset