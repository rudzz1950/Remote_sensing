import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import torch
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteSensingDetector:
    """
    A class for detecting buildings and trees in remote sensing images.
    Handles both single image and batch processing with custom class support.
    """
    
    # Custom class names for remote sensing
    CLASS_NAMES = {
        0: 'building',
        1: 'tree',
        2: 'road',
        3: 'water',
        4: 'vehicle'
    }
    
    # Color mapping for visualization (BGR format)
    CLASS_COLORS = {
        'building': (0, 0, 255),    # Red
        'tree': (0, 255, 0),       # Green
        'road': (128, 128, 128),   # Gray
        'water': (255, 0, 0),      # Blue
        'vehicle': (0, 255, 255)   # Yellow
    }
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the detector with a pre-trained model.
        
        Args:
            model_path: Path to the model weights file
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Default confidence thresholds
        self.conf_thresholds = {
            'building': 0.5,
            'tree': 0.4,
            'road': 0.3,
            'water': 0.4,
            'vehicle': 0.5
        }
        
        # Load the model
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the YOLO model with error handling."""
        try:
            logger.info(f"Loading model from {model_path}...")
            self.model = YOLO(model_path)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def process_image(
        self, 
        image: Union[str, np.ndarray, Path],
        output_dir: str = None,
        save: bool = True
    ) -> Dict:
        """
        Process a single image for object detection.
        
        Args:
            image: Path to the image or numpy array
            output_dir: Directory to save results
            save: Whether to save the results
            
        Returns:
            Dictionary containing detection results
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image in BGR format
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_name = Path(image_path).stem
        elif isinstance(image, np.ndarray):
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Input image must be a 3-channel RGB image")
            image_rgb = image
            image_name = "image"
        else:
            raise ValueError("image must be a string, Path, or numpy array")
        
        try:
            # Run inference
            results = self.model(
                image_rgb,
                conf=min(self.conf_thresholds.values()),  # Use minimum confidence threshold
                verbose=False
            )
            
            # Process results
            detections = self._process_detections(results[0], image_rgb.shape)
            
            # Save results if requested
            if save and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self._save_results(
                    image_rgb, 
                    detections, 
                    output_dir, 
                    image_name
                )
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def _process_detections(self, result, image_shape) -> Dict:
        """Process detection results into a structured format."""
        detections = {
            'boxes': [],
            'scores': [],
            'class_ids': [],
            'class_names': [],
            'masks': [] if hasattr(result, 'masks') else None
        }
        
        if result.boxes is not None:
            # Process bounding boxes
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            detections['boxes'] = boxes.tolist()
            detections['scores'] = scores.tolist()
            detections['class_ids'] = class_ids.tolist()
            detections['class_names'] = [self.CLASS_NAMES.get(cid, 'unknown') for cid in class_ids]
        
        # Process masks if available (for segmentation)
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            detections['masks'] = masks.tolist()
        
        return detections
    
    def _save_results(
        self, 
        image: np.ndarray, 
        detections: Dict, 
        output_dir: str, 
        base_name: str
    ) -> None:
        """Save detection results to files."""
        try:
            # Save visualization
            vis_image = self._visualize_detections(image, detections)
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to {output_path}")
            
            # Save detection data as text
            txt_path = os.path.join(output_dir, f"{base_name}_detections.txt")
            with open(txt_path, 'w') as f:
                f.write("Detection Results:\n")
                f.write("=" * 20 + "\n\n")
                
                if not detections['boxes']:
                    f.write("No objects detected.\n")
                else:
                    f.write(f"Detected {len(detections['boxes'])} objects:\n\n")
                    for i, (box, score, class_id, class_name) in enumerate(zip(
                        detections['boxes'],
                        detections['scores'],
                        detections['class_ids'],
                        detections['class_names']
                    )):
                        f.write(f"{i+1}. Class: {class_name} (ID: {class_id})\n")
                        f.write(f"   Confidence: {score:.4f}\n")
                        f.write(f"   Bounding Box: {[round(x, 2) for x in box]}\n\n")
            
            logger.info(f"Saved detection results to {txt_path}")
            
            # Save results as JSON
            import json
            import numpy as np
            
            def numpy_to_python(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: numpy_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [numpy_to_python(x) for x in obj]
                return obj
            
            json_path = os.path.join(output_dir, f"{base_name}_results.json")
            with open(json_path, 'w') as f:
                json.dump(numpy_to_python(detections), f, indent=2)
            
            logger.info(f"Saved detailed results to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _visualize_detections(
        self, 
        image: np.ndarray, 
        detections: Dict
    ) -> np.ndarray:
        """Draw detection results on the image."""
        # Create a copy of the image
        vis_image = image.copy()
        
        # Draw bounding boxes
        for box, class_name in zip(detections['boxes'], detections['class_names']):
            x1, y1, x2, y2 = map(int, box)
            color = self.CLASS_COLORS.get(class_name, (0, 0, 0))  # Default to black
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add class label
            label = f"{class_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(
                vis_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        # Draw masks if available
        if 'masks' in detections and detections['masks']:
            for mask_data in detections['masks']:
                # Convert mask to binary
                mask = np.array(mask_data) > 0.5  # Threshold at 0.5
                
                # Create a color mask
                color_mask = np.zeros_like(vis_image)
                color_mask[mask] = list(self.CLASS_COLORS.get('tree', (0, 255, 0)))
                
                # Blend with original image
                alpha = 0.3
                vis_image = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)
        
        return vis_image

# Example usage
if __name__ == "__main__":
    # Example usage
    detector = RemoteSensingDetector("yolov8n.pt")  # Replace with your model path
    
    # Process an image
    results = detector.process_image(
        "path/to/your/image.jpg",
        output_dir="outputs",
        save=True
    )
