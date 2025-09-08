import os
import cv2
import numpy as np
from pathlib import Path
import logging
from detector import RemoteSensingDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(output_path, size=(512, 512)):
    """Create a test image with simple shapes for buildings and trees."""
    # Create a white background
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Draw buildings (rectangles with different shades of gray)
    cv2.rectangle(img, (50, 50), (150, 200), (120, 120, 120), -1)  # Building 1
    cv2.rectangle(img, (200, 100), (300, 250), (100, 100, 100), -1)  # Building 2
    cv2.rectangle(img, (350, 150), (450, 300), (80, 80, 80), -1)    # Building 3
    
    # Draw trees (green circles with different sizes and shades)
    cv2.circle(img, (400, 150), 40, (0, 100, 0), -1)      # Tree 1
    cv2.circle(img, (100, 400), 50, (0, 120, 0), -1)      # Tree 2
    cv2.circle(img, (250, 400), 35, (0, 80, 0), -1)       # Tree 3
    cv2.circle(img, (400, 400), 45, (0, 140, 0), -1)      # Tree 4
    
    # Add some roads (gray lines)
    cv2.line(img, (0, 300), (512, 300), (150, 150, 150), 10)
    cv2.line(img, (250, 0), (250, 512), (150, 150, 150), 10)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info(f"Created test image at: {output_path}")
    return img

def main():
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test image
    test_image_path = os.path.join('test_images', 'remote_sensing_test.jpg')
    test_image = create_test_image(test_image_path)
    
    # Initialize the detector with PyTorch model paths
    model_paths = {
        'building_detection': 'models/building_detection.pt',  # PyTorch model for building detection
        'tree_segmentation': 'models/tree_segmentation.pt'  # PyTorch model for tree segmentation
    }
    
    try:
        logger.info("Initializing RemoteSensingDetector...")
        detector = RemoteSensingDetector(model_paths)
        
        # Process the test image
        logger.info("Processing test image...")
        results = detector.process_image(
            test_image,  # Use the numpy array directly
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
