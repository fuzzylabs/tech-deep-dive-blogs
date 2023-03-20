"""A step which preprocesses image data in a HuggingFace Dataset."""
from zenml.logger import get_logger
from zenml.steps import step
from datasets import Dataset
from transformers import AutoImageProcessor
from PIL import Image
import torch

logger = get_logger(__name__)


def preprocess_image(img: Image, image_processor: AutoImageProcessor) -> torch.tensor:
    """Preprocess image using a HuggingFace auto image processor.

    Args:
        img (Image): Pillow image
        image_processor (AutoImageProcessor): HuggingFace image processor

    Returns:
        torch.tensor: Preprocessed image as a Torch tensor
    """
    # Convert image to RGB if it is not already.
    img = img.convert("RGB")
    
    return image_processor(images = img, return_tensors = "pt")


@step
def preprocess_image_data(image_dataset: Dataset) -> Dataset:
    """Preprocess image data in a HuggingFace Dataset.
    
    Args:
        image_dataset (Dataset): A HuggingFace dataset containing the images as PIL objects.
    
    Returns:
        Dataset: A HuggingFace dataset containing a new column with preprocessed image tensors.
    """
    # Get image processor
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    
    # Process images using HuggingFace processor
    image_dataset = image_dataset.map(lambda x: {"preprocessed_image": preprocess_image(x['image'], image_processor=image_processor)})

    # Set dataset format to PyTorch
    image_dataset = image_dataset.with_format("pt")
    
    return image_dataset
