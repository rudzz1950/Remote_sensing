import os
import torch
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name: str, save_dir: str = 'models') -> str:
    """
    Download a pre-trained model from the Ultralytics model zoo.
    
    Args:
        model_name: Name of the model to download (e.g., 'yolov8n.pt')
        save_dir: Directory to save the downloaded model
        
    Returns:
        str: Path to the downloaded model
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        return model_path
    
    try:
        logger.info(f"Downloading {model_name}...")
        model = YOLO(f"{model_name}")
        
        # Save the model to the specified path
        model.export(format="onnx")
        os.rename(f"{model_name.split('.')[0]}.onnx", model_path)
        
        logger.info(f"Model downloaded and saved to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def main():
    # Download a model pre-trained on a dataset with buildings and trees
    # Note: In a real-world scenario, you would train or fine-tune a model on your specific dataset
    model_name = "yolov8n.pt"  # Using a small model for testing
    try:
        model_path = download_model(model_name)
        logger.info(f"Model is ready at: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
