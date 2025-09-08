"""
Script to download required YOLO models for building and tree detection.
"""
from ultralytics import YOLO
import os

def download_models():
    """Download required YOLO models."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Model URLs (using YOLOv8 models as they are well-supported by ultralytics)
    models = {
        'building_detection': 'yolov8n.pt',  # Using YOLOv8n as a starting point
        'tree_segmentation': 'yolov8n-seg.pt'  # Using YOLOv8n-seg for segmentation
    }
    
    # Download each model
    for model_name, model_id in models.items():
        print(f"Downloading {model_name}...")
        try:
            model = YOLO(model_id)
            # Save the model to the models directory
            model_path = os.path.join('models', f'{model_name}.pt')
            model.export(format='onnx')  # Export to ONNX for better compatibility
            print(f"Successfully downloaded and saved {model_name} to {model_path}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            
    print("\nModel download completed. You can now run the application.")

if __name__ == "__main__":
    download_models()
