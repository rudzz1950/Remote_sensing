import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from ultralytics import YOLO
import torch
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                        Expected keys: 'building_detection', 'tree_segmentation'
            device: Device to run the models on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
        # Default confidence thresholds
        self.conf_thresholds = {
            'building': 0.3,  # Default confidence threshold for building detection
            'tree': 0.25      # Default confidence threshold for tree segmentation
        }
        
        # Load models
        for model_name, model_path in model_paths.items():
            if model_path and os.path.exists(model_path):
                try:
                    # Determine task based on model name
                    task = 'segment' if 'seg' in model_name.lower() else 'detect'
                    
                    # Load the model with the appropriate task
                    model = YOLO(model_path, task=task)
                    
                    # Move to device if it's a PyTorch model
                    if not model.overrides.get('task') == 'segment':
                        model = model.to(self.device)
                    
                    self.models[model_name] = {
                        'model': model,
                        'task': task,
                        'is_onnx': False
                    }
                    
                    logger.info(f"Loaded model: {model_name} from {model_path} (task: {task}, device: {self.device})")
                    
                except Exception as e:
                    logger.error(f"Error loading model {model_name} from {model_path}: {str(e)}")
                    if 'No module named' in str(e):
                        logger.error("Make sure you have all required dependencies installed. Try running: pip install -r requirements.txt")
                    raise
        
        if not self.models:
            raise ValueError("No valid models were loaded. Please check the model paths.")
    
    def process_image(self, image_path: Union[str, np.ndarray, Path], 
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
        # Handle different input types
        if isinstance(image_path, (str, Path)):
            if not os.path.exists(str(image_path)):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_name = Path(image_path).stem
        elif isinstance(image_path, np.ndarray):
            if len(image_path.shape) != 3 or image_path.shape[2] != 3:
                raise ValueError("Input image must be a 3-channel RGB image")
            image = image_path
            image_name = "image"
        else:
            raise ValueError("image_path must be a string, Path, or numpy array")
        
        results = {}
        
        try:
            # Process with each model
            for model_name, model_info in self.models.items():
                try:
                    # Set confidence threshold based on model type
                    conf = self.conf_thresholds['building'] if 'building' in model_name else self.conf_thresholds['tree']
                    
                    # Get the model and run inference
                    model = model_info['model']
                    
                    # For segmentation models, ensure we're using the right task
                    if model_info['task'] == 'segment':
                        model_results = model.predict(
                            image,
                            conf=conf,
                            verbose=False,
                            device=self.device if torch.cuda.is_available() else None
                        )
                    else:
                        # For detection models
                        model_results = model(
                            image,
                            conf=conf,
                            verbose=False,
                            device=self.device if torch.cuda.is_available() else None
                        )
                    
                    # Store results
                    if model_results:
                        results[model_name] = model_results[0]  # Get first (and only) result
                    else:
                        logger.warning(f"No results returned from model: {model_name}")
                        
                except Exception as e:
                    logger.error(f"Error processing image with model {model_name}: {str(e)}")
                    continue
            
            if not results:
                raise RuntimeError("No models successfully processed the image")
            
            # Combine results from different models
            combined = self._combine_results(results, image)
            
            # Save results if requested
            if save and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self._save_results(combined, output_dir, image_name)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {str(e)}")
            raise
    
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
        """
        Combine results from different models.
        
        Args:
            results: Dictionary of model results
            image: Original input image
            
        Returns:
            Dictionary containing combined results with detections and masks
        """
        combined = {
            'image': image,
            'detections': {},
            'masks': {},
            'model_info': {}
        }
        
        for model_name, result in results.items():
            model_info = {
                'task': 'segmentation' if 'seg' in model_name.lower() else 'detection',
                'classes': getattr(result, 'names', {0: model_name.split('_')[0]}),
                'conf_threshold': self.conf_thresholds['building' if 'building' in model_name else 'tree']
            }
            combined['model_info'][model_name] = model_info
            
            # Handle detection results
            if hasattr(result, 'boxes') and result.boxes is not None:
                try:
                    combined['detections'][model_name] = {
                        'boxes': result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy,
                        'scores': result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf,
                        'class_ids': result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls.astype(int),
                        'class_names': [model_info['classes'].get(int(cls_id), str(cls_id)) 
                                     for cls_id in (result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') 
                                                   else result.boxes.cls.astype(int))]
                    }
                except Exception as e:
                    logger.error(f"Error processing detection results from {model_name}: {str(e)}")
            
            # Handle segmentation masks
            if hasattr(result, 'masks') and result.masks is not None:
                try:
                    mask_data = result.masks.data
                    if hasattr(mask_data, 'cpu'):
                        mask_data = mask_data.cpu().numpy()
                    
                    combined['masks'][model_name] = {
                        'mask': mask_data[0] if isinstance(mask_data, (list, tuple)) else mask_data,
                        'class_id': 0,  # Default class ID for segmentation
                        'class_name': model_info['classes'].get(0, 'object')
                    }
                except Exception as e:
                    logger.error(f"Error processing segmentation masks from {model_name}: {str(e)}")
        
        return combined
    
    def _save_results(self, results: Dict, output_dir: str, base_name: str):
        """
        Save detection results to files.
        
        Args:
            results: Combined results from _combine_results
            output_dir: Directory to save output files
            base_name: Base name for output files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save visualization
            vis_image = self._visualize_results(results)
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to {output_path}")
            
            # Save detection data as text
            txt_path = os.path.join(output_dir, f"{base_name}_detections.txt")
            with open(txt_path, 'w') as f:
                # Write model information
                f.write(f"Model Information:\n{'='*20}\n")
                for model_name, info in results.get('model_info', {}).items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Task: {info['task']}\n")
                    f.write(f"Classes: {info['classes']}\n")
                    f.write(f"Confidence Threshold: {info['conf_threshold']}\n")
                    f.write("\n")
                
                # Write detection results
                if results.get('detections'):
                    f.write("\nDetection Results:\n" + "="*20 + "\n")
                    for model_name, dets in results['detections'].items():
                        if len(dets['boxes']) > 0:
                            f.write(f"\n{model_name} detected {len(dets['boxes'])} objects:\n")
                            for i, (box, score, cls_id, cls_name) in enumerate(zip(
                                dets['boxes'], dets['scores'], 
                                dets['class_ids'], dets.get('class_names', []))):
                                f.write(f"  {i+1}. Class: {cls_name} (ID: {cls_id}), "
                                      f"Confidence: {score:.2f}, "
                                      f"BBox: {box}\n")
                
                # Write segmentation results
                if results.get('masks'):
                    f.write("\nSegmentation Results:\n" + "="*20 + "\n")
                    for model_name, mask_data in results['masks'].items():
                        f.write(f"\n{model_name} generated segmentation mask "
                              f"for class: {mask_data.get('class_name', 'unknown')}\n")
            
            logger.info(f"Saved detection results to {txt_path}")
            
            # Save results as JSON
            try:
                import json
                import numpy as np
                
                def numpy_to_python(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
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
                    json.dump(numpy_to_python(results), f, indent=2)
                logger.info(f"Saved detailed results to {json_path}")
                
            except Exception as e:
                logger.warning(f"Could not save JSON results: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _visualize_results(self, results: Dict) -> np.ndarray:
        """
        Create a visualization of the detection results.
        
        Args:
            results: Dictionary containing detection and segmentation results
            
        Returns:
            np.ndarray: Visualized image with detections and masks
        """
        # Create a copy of the original image
        image = results['image'].copy()
        
        # Ensure the image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw bounding boxes
        for model_name, detections in results.get('detections', {}).items():
            if not all(k in detections for k in ['boxes', 'scores', 'class_ids']):
                logger.warning(f"Missing required keys in detections for {model_name}")
                continue
                
            for box, score, class_id in zip(detections['boxes'], 
                                          detections['scores'], 
                                          detections['class_ids']):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Different colors for different models
                    if 'building' in model_name.lower():
                        color = (255, 0, 0)  # Red for buildings
                    else:
                        color = (0, 255, 0)  # Green for trees
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{model_name.split('_')[0]} {float(score):.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    logger.error(f"Error drawing detection for {model_name}: {str(e)}")
        
        # Draw masks
        for model_name, mask_data in results.get('masks', {}).items():
            if 'mask' not in mask_data:
                logger.warning(f"No mask data found for {model_name}")
                continue
                
            try:
                mask = mask_data['mask']
                
                # Create a color mask based on model type
                if 'building' in model_name.lower():
                    color = np.array([255, 0, 0], dtype=np.uint8)  # Red for buildings
                else:
                    color = np.array([0, 255, 0], dtype=np.uint8)  # Green for trees
                
                # Ensure mask is 2D
                if len(mask.shape) == 3 and mask.shape[0] == 1:
                    mask = mask[0]  # Remove single-channel dimension if present
                
                # Resize mask to match image dimensions if needed
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.float32), 
                                    (image.shape[1], image.shape[0]))
                
                # Create a colored mask
                mask_binary = (mask > 0.5).astype(np.uint8)
                mask_colored = np.zeros_like(image, dtype=np.uint8)
                mask_colored[mask_binary > 0] = color
                
                # Apply mask with transparency
                alpha = 0.3
                mask_alpha = (mask * alpha).astype(np.float32)
                if len(mask_alpha.shape) == 2:
                    mask_alpha = np.repeat(mask_alpha[..., np.newaxis], 3, axis=2)
                
                # Blend the colored mask with the original image
                image = cv2.addWeighted(
                    image.astype(np.float32), 1.0,
                    mask_colored.astype(np.float32), 0.3,
                    0
                ).astype(np.uint8)
                
            except Exception as e:
                logger.error(f"Error processing mask for {model_name}: {str(e)}")
                continue
        
        return image
