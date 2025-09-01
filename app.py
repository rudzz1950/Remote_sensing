#!/usr/bin/env python3
"""
Remote Sensing - Building and Tree Detection

This is the main entry point for the Remote Sensing application.
It provides both command-line and web interfaces for detecting buildings and trees
in aerial/satellite imagery.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional

from src.detector import RemoteSensingDetector
from src.web_interface import run_web_ui

# Default model paths (update these with your actual model paths)
DEFAULT_MODEL_PATHS = {
    'building_detection': 'models/building_detection.pt',
    'building_segmentation': 'models/building_segmentation.pt',
    'tree_segmentation': 'models/tree_segmentation.pt'
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect buildings and trees in aerial/satellite imagery.'
    )
    
    # Input/Output
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input image, video, or directory of images',
        required=False
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs/)'
    )
    
    # Model options
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing model files (default: models/)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (e.g., "cuda:0", "cpu"). Auto-detects if not specified.'
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--web',
        action='store_true',
        help='Launch web interface'
    )
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images in batch mode'
    )
    
    # Web interface options
    web_group = parser.add_argument_group('Web Interface Options')
    web_group.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to run the web server on (default: 0.0.0.0)'
    )
    web_group.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the web server on (default: 7860)'
    )
    web_group.add_argument(
        '--share',
        action='store_true',
        help='Create a public link for the web interface'
    )
    
    return parser.parse_args()

def ensure_model_paths(model_dir: str) -> Dict[str, str]:
    """Ensure model paths exist and return a dictionary of valid paths."""
    model_paths = {}
    model_dir = Path(model_dir)
    
    for model_name, default_path in DEFAULT_MODEL_PATHS.items():
        # Check if default path exists
        if os.path.exists(default_path):
            model_paths[model_name] = str(default_path)
            continue
            
        # Check in model directory
        model_path = model_dir / f"{model_name}.pt"
        if model_path.exists():
            model_paths[model_name] = str(model_path)
    
    if not model_paths:
        raise FileNotFoundError(
            f"No model files found in {model_dir}. "
            "Please ensure model files are present or specify --model-dir."
        )
        
    return model_paths

def main():
    """Main entry point."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Get model paths
        model_paths = ensure_model_paths(args.model_dir)
        
        # Initialize detector
        detector = RemoteSensingDetector(
            model_paths=model_paths,
            device=args.device
        )
        
        # Set confidence threshold
        detector.conf_thresholds = {
            'building': args.conf,
            'tree': args.conf
        }
        
        # Run in appropriate mode
        if args.web:
            # Web interface mode
            run_web_ui(
                model_paths=model_paths,
                host=args.host,
                port=args.port,
                share=args.share
            )
        elif args.batch or (args.input and os.path.isdir(args.input)):
            # Batch processing mode
            if not args.input:
                raise ValueError("Input directory must be specified for batch processing")
                
            print(f"Processing images in: {args.input}")
            detector.process_batch(
                input_path=args.input,
                output_dir=args.output
            )
            print(f"Results saved to: {os.path.abspath(args.output)}")
            
        elif args.input:
            # Single file processing
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
                
            print(f"Processing: {args.input}")
            detector.process_image(
                image_path=args.input,
                output_dir=args.output
            )
            print(f"Results saved to: {os.path.abspath(args.output)}")
            
        else:
            # No mode specified, show help
            print("No mode specified. Use --web for web interface or provide an input file/directory.")
            print("Use --help for usage information.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.web:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
