import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
from ultralytics import YOLO
import torch
from tqdm import tqdm

class RemoteSensingDetector:
    """
    A class for detecting buildings and trees in remote sensing images.
    Handles both single image and batch processing.
    """
    
    def __init__(self, model_paths: Dict[str, str], device: str = None):
        """
        Initialize the detector with pre-trained models.
        
        Args:
            model_paths: Dictionary containing paths to model weights
                        Expected keys: 'building_detection', 'building_segmentation', 'tree_segmentation'
            device: Device to run the models on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
        # Load models
        for model_name, model_path in model_paths.items():
            if model_path and os.path.exists(model_path):
                self.models[model_name] = YOLO(model_path).to(self.device)
        
        # Default confidence thresholds
        self.conf_thresholds = {
            'building': 0.3,
            'tree': 0.25
        }
    
    def process_image(self, image_path: Union[str, np.ndarray], 
                     output_dir: str = None, save: bool = True) -> Dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image or numpy array
            output_dir: Directory to save results
            save: Whether to save the results
            
        Returns:
            Dictionary containing detection results
        """
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        results = {}
        
        # Process with each model
        for model_name, model in self.models.items():
            if 'building' in model_name:
                conf = self.conf_thresholds['building']
            else:
                conf = self.conf_thresholds['tree']
                
            model_results = model(image, conf=conf)
            results[model_name] = model_results[0]  # Get first (and only) result
        
        # Combine results
        combined = self._combine_results(results, image)
        
        if save and output_dir:
            self._save_results(combined, output_dir, Path(image_path).stem)
        
        return combined
    
    def process_batch(self, input_path: str, output_dir: str, 
                     extensions: List[str] = None) -> List[Dict]:
        """
        Process multiple images in a directory.
        
        Args:
            input_path: Path to input directory or file
            output_dir: Directory to save results
            extensions: List of file extensions to process (e.g., ['.jpg', '.png'])
            
        Returns:
            List of results for each processed image
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            
        input_path = Path(input_path)
        image_paths = []
        
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            image_paths = [p for p in input_path.glob('*') 
                          if p.suffix.lower() in extensions]
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.process_image(
                    str(img_path), 
                    output_dir=output_dir,
                    save=True
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
                
        return results
    
    def _combine_results(self, results: Dict, image: np.ndarray) -> Dict:
        """Combine results from different models."""
        combined = {
            'image': image,
            'detections': {},
            'masks': {}
        }
        
        for model_name, result in results.items():
            if hasattr(result, 'boxes') and result.boxes is not None:
                combined['detections'][model_name] = {
                    'boxes': result.boxes.xyxy.cpu().numpy(),
                    'scores': result.boxes.conf.cpu().numpy(),
                    'class_ids': result.boxes.cls.cpu().numpy().astype(int)
                }
                
            if hasattr(result, 'masks') and result.masks is not None:
                combined['masks'][model_name] = {
                    'mask': result.masks.data.cpu().numpy(),
                    'class_id': 0  # Default class ID for segmentation
                }
        
        return combined
    
    def _save_results(self, results: Dict, output_dir: str, base_name: str):
        """Save detection results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        vis_image = self._visualize_results(results)
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Save detection data
        # TODO: Add support for GeoJSON export
    
    def _visualize_results(self, results: Dict) -> np.ndarray:
        """Create a visualization of the detection results."""
        image = results['image'].copy()
        
        # Draw bounding boxes
        for model_name, detections in results['detections'].items():
            for box, score, class_id in zip(detections['boxes'], 
                                          detections['scores'], 
                                          detections['class_ids']):
                x1, y1, x2, y2 = map(int, box)
                
                # Different colors for different models
                if 'building' in model_name:
                    color = (255, 0, 0)  # Red for buildings
                else:
                    color = (0, 255, 0)  # Green for trees
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{model_name.split('_')[0]} {score:.2f}"
                cv2.putText(image, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw masks
        for model_name, mask_data in results['masks'].items():
            mask = mask_data['mask']
            class_id = mask_data['class_id']
            
            # Create a color mask
            if 'building' in model_name:
                color = (255, 0, 0)  # Red for buildings
            else:
                color = (0, 255, 0)  # Green for trees
            
            # Apply mask with transparency
            mask = (mask[0] * 255).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask = (mask * np.array(color) / 255).astype(np.uint8)
            
            # Blend with original image
            image = cv2.addWeighted(image, 1.0, mask, 0.3, 0)
        
        return image
