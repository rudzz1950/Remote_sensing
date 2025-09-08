import os
import cv2
import numpy as np
from detector import RemoteSensingDetector
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Define model paths
    model_paths = {
        'building_detection': 'models/yolov8n.onnx',
        'tree_segmentation': 'models/yolov8n-seg.onnx'
    }
    
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize the detector
        logger.info("Initializing RemoteSensingDetector...")
        detector = RemoteSensingDetector(model_paths)
        
        # Create a test image (white background with some colored shapes)
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Draw some test shapes (rectangles and circles)
        cv2.rectangle(test_image, (50, 50), (200, 200), (0, 0, 255), -1)  # Red rectangle (building)
        cv2.circle(test_image, (400, 150), 80, (0, 255, 0), -1)  # Green circle (tree)
        cv2.rectangle(test_image, (300, 300), (450, 450), (0, 0, 255), -1)  # Another red rectangle (building)
        
        # Save the test image
        test_image_path = os.path.join('test_images', 'test_image.jpg')
        cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Created test image at: {test_image_path}")
        
        # Process the test image
        logger.info("Processing test image...")
        results = detector.process_image(
            test_image,  # Use the numpy array directly
            output_dir=output_dir,
            save=True
        )
        
        logger.info("Processing completed successfully!")
        logger.info(f"Results saved in: {os.path.abspath(output_dir)}")
        
        # Display the output image
        output_image_path = os.path.join(output_dir, 'test_image_result.jpg')
        if os.path.exists(output_image_path):
            logger.info(f"You can view the results at: {os.path.abspath(output_image_path)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
