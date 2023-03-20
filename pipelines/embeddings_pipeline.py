"""A pipeline to create embeddings for artwork."""
from zenml.logger import get_logger
from zenml.pipelines import pipeline

logger = get_logger(__name__)


@pipeline
def embeddings_pipeline(
    import_data,
    preprocess_image_data,
    create_embeddings
):
    """NFT Embedding pipeline.

    Args:
        import_data: a step to get the image data from the data folder
        preprocess_image_dataset: a step to preprocess the image data
        create_embeddings: a step to create the embeddings and put them into the HuggingFace dataset
    """
    # Create the dataset
    image_dataset = import_data()
    
    # Preprocess image data
    image_dataset = preprocess_image_data(image_dataset)
    
    # Extract the embeddings
    image_dataset = create_embeddings(image_dataset)