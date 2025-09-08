"""
Project Setup Script

This script sets up the project structure and downloads required models.
"""
import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure."""
    print("Creating directory structure...")
    
    # Define directories to create
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "outputs"
    ]
    
    # Create each directory
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def download_models():
    """Download required YOLO models."""
    print("\nDownloading models...")
    try:
        from ultralytics import YOLO
        import shutil
        
        # Model configurations
        models = [
            {"name": "yolov8n.pt", "url": "yolov8n.pt"},  # Base detection model
            {"name": "yolov8n-seg.pt", "url": "yolov8n-seg.pt"},  # Segmentation model
        ]
        
        for model_info in models:
            model_name = model_info["name"]
            model_path = os.path.join("models", model_name)
            
            if not os.path.exists(model_path):
                print(f"Downloading {model_name}...")
                try:
                    # Download the model
                    model = YOLO(model_info["url"])
                    
                    # Save the model
                    output_path = model.export(format="onnx")
                    
                    # Move the exported model to the models directory
                    if os.path.exists(output_path):
                        shutil.move(output_path, os.path.join("models", os.path.basename(output_path)))
                        print(f"Successfully downloaded and saved {model_name}")
                    else:
                        print(f"Warning: Could not find exported model at {output_path}")
                        
                except Exception as e:
                    print(f"Error downloading {model_name}: {str(e)}")
            else:
                print(f"{model_name} already exists, skipping download.")
                
    except ImportError:
        print("Ultralytics package not found. Please install it first using: pip install ultralytics")
        sys.exit(1)

def main():
    """Main function to set up the project."""
    print("=" * 50)
    print("Remote Sensing Project Setup")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    install_dependencies()
    
    # Download models
    download_models()
    
    print("\nSetup completed successfully!")
    print("You can now run the application using: python app.py --web")

if __name__ == "__main__":
    main()
