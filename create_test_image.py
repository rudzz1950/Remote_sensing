import cv2
import numpy as np
import os

def create_test_image(output_path='test_images/objects_test.jpg', size=(800, 800)):
    """Create a test image with common objects for detection."""
    # Create a white background
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Draw some common objects that YOLOv8 can detect
    
    # Draw a car (rectangle with circles for wheels)
    cv2.rectangle(img, (100, 500), (300, 400), (0, 0, 255), -1)  # Car body
    cv2.circle(img, (150, 530), 20, (0, 0, 0), -1)  # Wheel 1
    cv2.circle(img, (250, 530), 20, (0, 0, 0), -1)  # Wheel 2
    
    # Draw a person (stick figure)
    cv2.circle(img, (500, 300), 30, (0, 255, 0), -1)  # Head
    cv2.line(img, (500, 330), (500, 400), (0, 255, 0), 5)  # Body
    cv2.line(img, (500, 350), (450, 300), (0, 255, 0), 5)  # Left arm
    cv2.line(img, (500, 350), (550, 300), (0, 255, 0), 5)  # Right arm
    cv2.line(img, (500, 400), (450, 450), (0, 255, 0), 5)  # Left leg
    cv2.line(img, (500, 400), (550, 450), (0, 255, 0), 5)  # Right leg
    
    # Draw a dog (simplified)
    cv2.ellipse(img, (300, 200), (80, 40), 0, 0, 360, (255, 0, 0), -1)  # Body
    cv2.circle(img, (350, 180), 20, (255, 0, 0), -1)  # Head
    cv2.circle(img, (360, 175), 3, (0, 0, 0), -1)  # Eye
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Created test image at: {output_path}")
    return img

if __name__ == "__main__":
    create_test_image()
