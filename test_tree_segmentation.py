import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path

def test_tree_segmentation(image_path, model_path='models/tree_segmentation.pt', conf=0.25):
    """
    Test tree segmentation model on an image with adjustable confidence threshold
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"\nTesting tree segmentation on {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Run inference
    results = model(image_rgb, conf=conf)
    
    # Process results
    for i, r in enumerate(results):
        # Print class names if available
        if hasattr(r, 'names') and r.names is not None:
            print(f"\nAvailable classes: {r.names}")
        
        # Print detections
        if hasattr(r, 'boxes') and r.boxes is not None:
            print(f"\nDetected {len(r.boxes)} objects:")
            for j, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                conf = box.conf[0].item()
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = r.names[cls] if hasattr(r, 'names') and r.names is not None else str(cls)
                print(f"  {j+1}. {cls_name} ({conf:.2f}): {xyxy}")
        else:
            print("No objects detected.")
    
    # Save visualization
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{Path(image_path).stem}_tree_segmentation.jpg")
    
    # Plot results
    plotted = results[0].plot()
    cv2.imwrite(output_path, cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))
    print(f"\nSaved visualization to {output_path}")

if __name__ == "__main__":
    # Test on our remote sensing image
    test_image = "test_images/remote_sensing_test.jpg"
    
    # Test with tree segmentation model
    if os.path.exists("models/tree_segmentation.pt"):
        test_tree_segmentation(test_image, conf=0.1)  # Lower confidence threshold
    else:
        print("Tree segmentation model not found.")
