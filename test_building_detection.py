import cv2
import os
from pathlib import Path
from remote_sensing_detector import RemoteSensingDetector

def main():
    # Initialize the detector with a lower confidence threshold
    detector = RemoteSensingDetector(
        model_path='models/building_detection.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Set lower confidence thresholds for all classes
    detector.conf_thresholds = {
        'building': 0.1,  # Lower threshold for buildings
        'tree': 0.1,      # Lower threshold for trees
        'road': 0.1,      # Lower threshold for roads
        'water': 0.1,     # Lower threshold for water
        'vehicle': 0.1    # Lower threshold for vehicles
    }
    
    # Process the test image
    image_path = 'test_images/remote_sensing_test.jpg'
    output_dir = 'outputs'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the image
    results = detector.process_image(image_path, output_dir=output_dir, save=True)
    
    # Print detection results
    print("\nDetection Results:")
    if results and 'detections' in results and results['detections']:
        for i, det in enumerate(results['detections'], 1):
            print(f"\nDetection {i}:")
            print(f"  Class: {det['label']} (ID: {det['label_id']})")
            print(f"  Confidence: {det['score']:.4f}")
            print(f"  Bounding Box: {det['box']}")
    else:
        print("No objects detected.")

if __name__ == "__main__":
    import torch  # Import torch here to avoid circular imports
    main()
