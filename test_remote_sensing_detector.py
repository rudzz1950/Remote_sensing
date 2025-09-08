import os
import cv2
import numpy as np
import logging
from pathlib import Path
from remote_sensing_detector import RemoteSensingDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(output_path, size=(800, 800)):
    """Create a test image with buildings and trees."""
    # Create a white background
    img = np.ones((*size, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw buildings (rectangles with different shades of gray)
    cv2.rectangle(img, (50, 50), (200, 300), (80, 80, 80), -1)    # Building 1
    cv2.rectangle(img, (250, 100), (400, 350), (100, 100, 100), -1)  # Building 2
    cv2.rectangle(img, (450, 150), (600, 400), (120, 120, 120), -1)  # Building 3
    
    # Draw trees (green circles with different sizes and shades)
    cv2.circle(img, (150, 500), 60, (0, 100, 0), -1)      # Tree 1
    cv2.circle(img, (300, 450), 40, (0, 120, 0), -1)      # Tree 2
    cv2.circle(img, (400, 550), 50, (0, 80, 0), -1)       # Tree 3
    cv2.circle(img, (600, 500), 55, (0, 140, 0), -1)      # Tree 4
    
    # Add roads (gray lines)
    cv2.line(img, (0, 400),  (800, 400),  (150, 150, 150), 15)  # Horizontal road
    cv2.line(img, (400, 0),  (400, 800),  (150, 150, 150), 15)  # Vertical road
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    logger.info(f"Created test image at: {output_path}")
    return img

def main():
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test image
    test_image_path = os.path.join('test_images', 'remote_sensing_test.jpg')
    test_image = create_test_image(test_image_path)
    
    # Convert BGR to RGB (our detector expects RGB)
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    try:
        # Initialize the detector with a model
        # Note: In a real application, you would use a model trained on remote sensing data
        model_path = "yolov8n.pt"  # Replace with your actual model path
        logger.info(f"Initializing RemoteSensingDetector with model: {model_path}")
        detector = RemoteSensingDetector(model_path)
        
        # Process the test image
        logger.info("Processing test image...")
        results = detector.process_image(
            test_image_rgb,
            output_dir=output_dir,
            save=True
        )
        
        logger.info("Processing completed successfully!")
        logger.info(f"Results saved in: {os.path.abspath(output_dir)}")
        
        # Display the output image path
        output_image_path = os.path.join(output_dir, 'remote_sensing_test_result.jpg')
        if os.path.exists(output_image_path):
            logger.info(f"You can view the results at: {os.path.abspath(output_image_path)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
