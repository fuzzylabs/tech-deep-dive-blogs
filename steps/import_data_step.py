"""A step which imports the image data from the data folder."""
from zenml.logger import get_logger
from zenml.steps import step
from datasets import load_dataset, Dataset
from torchvision.datasets.folder import ImageFolder

logger = get_logger(__name__)


@step
def import_data() -> Dataset:
    """Imports the image data from the local folder.

    Returns:
        Dataset: A HuggingFace dataset containing the images as PIL objects.
    """
    image_dataset = load_dataset("imagefolder", data_dir="data/")

    # Remove train/test segmentation
    image_dataset = image_dataset['train']
    
    return image_dataset
