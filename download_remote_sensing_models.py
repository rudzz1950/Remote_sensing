import os
import urllib.request
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs (replace with actual remote sensing model URLs)
MODEL_URLS = {
    'building_detection': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'tree_segmentation': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt'
}

def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL to the specified path."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Downloading {url} to {output_path}")
        
        # Show progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, int(downloaded * 100 / total_size))
                print(f"\rDownloading... {percent}%", end='')
        
        with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
        print("\nDownload completed!")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download each model
    for model_name, url in MODEL_URLS.items():
        output_path = models_dir / f"{model_name}.pt"
        
        # Skip if model already exists
        if output_path.exists():
            logger.info(f"Model {model_name} already exists at {output_path}")
            continue
            
        logger.info(f"Downloading {model_name} model...")
        success = download_file(url, str(output_path))
        
        if success:
            logger.info(f"Successfully downloaded {model_name} model to {output_path}")
        else:
            logger.error(f"Failed to download {model_name} model")
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main()
