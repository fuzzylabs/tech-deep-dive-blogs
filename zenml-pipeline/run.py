from pipelines.embeddings_pipeline import embeddings_pipeline
from steps.create_embeddings_step import create_embeddings
from steps.import_data_step import import_data
from steps.preprocess_image_data_step import preprocess_image_data

def main():
    """Run the ZenML pipeline."""
    embeddings_pipeline(
        import_data(),
        preprocess_image_data(),
        create_embeddings()
    ).run()
    
if __name__ == "__main__":
    main()