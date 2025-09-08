import os
import cv2
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteSensingDetectorHF:
    """
    A class for detecting buildings in remote sensing images using Hugging Face models.
    """
    
    def __init__(self, model_path: str = "models/building_detection.pt"):  # Using custom building detection model
        """
        Initialize the detector with a YOLOv8 model.
        
        Args:
            model_path: Path to the YOLOv8 model file
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Define class names for YOLOv8 (COCO dataset classes)
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
            34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
            37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
            41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
            66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
        try:
            logger.info(f"Loading YOLOv8 model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def process_image(
        self, 
        image: Union[str, np.ndarray, Path],
        confidence_threshold: float = 0.25,  # Lower threshold to get more detections
        output_dir: str = "outputs",
        save: bool = True
    ) -> Dict:
        """
        Process a single image for object detection.
        
        Args:
            image: Path to the image or numpy array (BGR format)
            confidence_threshold: Minimum confidence score for detections
            output_dir: Directory to save results
            save: Whether to save the results
            
        Returns:
            Dictionary containing detection results
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_name = image_path.name
            image_rgb = cv2.imread(str(image_path))
            if image_rgb is None:
                raise ValueError(f"Could not read image: {image_path}")
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_name = f"result_{int(time.time())}.jpg"
        else:
            raise ValueError("image must be a string, Path, or numpy array")
        
        try:
            # Run YOLOv8 inference
            results = self.model(image_rgb, conf=confidence_threshold)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()
                    cls_id = int(box.cls.item())
                    
                    # Get class name
                    label_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],  # x1, y1, x2, y2
                        'score': float(conf),
                        'label': label_name,
                        'label_id': cls_id
                    })
            
            # Save results if requested
            if save and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self._save_results(
                    image_rgb, 
                    detections, 
                    output_dir, 
                    image_name
                )
            
            return {
                'detections': detections,
                'image_size': image_rgb.shape[:2]
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def _save_results(
        self, 
        image: np.ndarray, 
        detections: List[Dict], 
        output_dir: str, 
        base_name: str
    ) -> None:
        """Save detection results to files."""
        try:
            # Create a copy of the image for visualization
            vis_image = image.copy()
            
            # Draw detections
            for det in detections:
                box = det['box']
                label = f"{det['label']} {det['score']:.2f}"
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(
                    vis_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
            
            # Save visualization
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to {output_path}")
            
            # Save detection data as text
            txt_path = os.path.join(output_dir, f"{base_name}_detections.txt")
            with open(txt_path, 'w') as f:
                f.write("Detection Results:\n")
                f.write("=" * 20 + "\n\n")
                
                if not detections:
                    f.write("No objects detected.\n")
                else:
                    f.write(f"Detected {len(detections)} objects:\n\n")
                    for i, det in enumerate(detections, 1):
                        f.write(f"{i}. Label: {det['label']} (ID: {det['label_id']})\n")
                        f.write(f"   Confidence: {det['score']:.4f}\n")
                        f.write(f"   Bounding Box: {[int(x) for x in det['box']]}\n\n")
            
            logger.info(f"Saved detection results to {txt_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run object detection on an image using YOLOv8')
    parser.add_argument('image_path', type=str, nargs='?', default='test_images/objects_test.jpg',
                       help='Path to the input image (default: test_images/objects_test.jpg)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = RemoteSensingDetectorHF()
    
    # Process the input image
    if os.path.exists(args.image_path):
        print(f"Processing image: {args.image_path}")
        start_time = time.time()
        
        # Process the image
        results = detector.process_image(
            image=args.image_path,
            confidence_threshold=args.conf,
            output_dir=args.output_dir,
            save=True
        )
        
        # Print results
        elapsed = time.time() - start_time
        print(f"Processing time: {elapsed:.2f} seconds")
        print(f"Detected {len(results['detections'])} objects")
        
        # Print detection details
        for i, det in enumerate(results['detections'], 1):
            print(f"\nDetection {i}:")
            print(f"  Label: {det['label']} (ID: {det['label_id']})")
            print(f"  Confidence: {det['score']:.4f}")
            print(f"  Bounding box: {det['box']}")
            
    else:
        print(f"Error: Image not found at {args.image_path}")
        print("Please provide a valid image path.")
