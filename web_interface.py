import gradio as gr
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import tempfile
from detector import RemoteSensingDetector

class RemoteSensingWebUI:
    """Gradio-based web interface for the Remote Sensing Detector."""
    
    def __init__(self, detector: RemoteSensingDetector):
        """
        Initialize the web interface.
        
        Args:
            detector: Initialized RemoteSensingDetector instance
        """
        self.detector = detector
        self.temp_dir = tempfile.mkdtemp(prefix="remote_sensing_")
        
    def create_interface(self):
        """Create and return the Gradio interface."""
        with gr.Blocks(title="Remote Sensing Detector") as demo:
            gr.Markdown("""
            # 🌍 Remote Sensing Detector
            Upload images or videos to detect buildings and trees using AI.
            """)
            
            with gr.Tabs():
                with gr.TabItem("Single Image"):
                    with gr.Row():
                        with gr.Column():
                            img_input = gr.Image(type="filepath", label="Upload Image")
                            with gr.Row():
                                submit_btn = gr.Button("Process", variant="primary")
                                clear_btn = gr.Button("Clear")
                            
                            # Model settings
                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    building_conf = gr.Slider(
                                        minimum=0.1, 
                                        maximum=1.0, 
                                        value=0.3, 
                                        step=0.05,
                                        label="Building Confidence Threshold"
                                    )
                                    tree_conf = gr.Slider(
                                        minimum=0.1, 
                                        maximum=1.0, 
                                        value=0.25, 
                                        step=0.05,
                                        label="Tree Confidence Threshold"
                                    )
                        
                        with gr.Column():
                            img_output = gr.Image(label="Detection Results")
                            download_btn = gr.Button("Download Results")
                
                with gr.TabItem("Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_input = gr.File(
                                file_count="multiple",
                                file_types=["image"],
                                label="Upload Images"
                            )
                            batch_dir = gr.Textbox(
                                label="Or enter directory path",
                                placeholder="Path to directory containing images"
                            )
                            with gr.Row():
                                batch_submit = gr.Button("Process Batch", variant="primary")
                                batch_clear = gr.Button("Clear")
                            
                            batch_progress = gr.Slider(
                                visible=False,
                                interactive=False,
                                label="Progress"
                            )
                        
                        with gr.Column():
                            batch_gallery = gr.Gallery(
                                label="Processed Images",
                                show_label=True,
                                columns=[3],
                                object_fit="contain",
                                height="auto"
                            )
                            batch_output_dir = gr.Textbox(
                                label="Output Directory",
                                value=os.path.join(os.getcwd(), "outputs"),
                                interactive=True
                            )
            
            # Single image processing
            submit_btn.click(
                fn=self.process_single_image,
                inputs=[img_input, building_conf, tree_conf],
                outputs=img_output
            )
            
            clear_btn.click(
                fn=lambda: [None, None],
                inputs=None,
                outputs=[img_input, img_output]
            )
            
            # Batch processing
            batch_submit.click(
                fn=self.process_batch_images,
                inputs=[batch_input, batch_dir, batch_output_dir, building_conf, tree_conf],
                outputs=[batch_gallery, batch_progress]
            )
            
            batch_clear.click(
                fn=lambda: [[], ""],
                inputs=None,
                outputs=[batch_gallery, batch_dir]
            )
            
            # Download handler
            download_btn.click(
                fn=self.download_results,
                inputs=None,
                outputs=gr.File(label="Download Results")
            )
            
        return demo
    
    def process_single_image(self, image_path: str, 
                           building_conf: float, 
                           tree_conf: float) -> np.ndarray:
        """Process a single uploaded image."""
        if image_path is None:
            return None
            
        # Update confidence thresholds
        self.detector.conf_thresholds['building'] = building_conf
        self.detector.conf_thresholds['tree'] = tree_conf
        
        # Process the image
        result = self.detector.process_image(
            image_path=image_path,
            output_dir=self.temp_dir,
            save=True
        )
        
        return result['image']
    
    def process_batch_images(self, files: List[str], 
                           input_dir: str,
                           output_dir: str,
                           building_conf: float,
                           tree_conf: float) -> List[str]:
        """Process multiple images in batch mode."""
        if not files and not input_dir:
            return [], 0
            
        # Update confidence thresholds
        self.detector.conf_thresholds['building'] = building_conf
        self.detector.conf_thresholds['tree'] = tree_conf
        
        # Process files
        processed_paths = []
        
        if files:
            # Process uploaded files
            for file in files:
                result = self.detector.process_image(
                    image_path=file.name,
                    output_dir=output_dir,
                    save=True
                )
                output_path = os.path.join(
                    output_dir, 
                    f"{Path(file.name).stem}_result.jpg"
                )
                processed_paths.append(output_path)
        elif input_dir and os.path.isdir(input_dir):
            # Process directory
            results = self.detector.process_batch(
                input_path=input_dir,
                output_dir=output_dir
            )
            processed_paths = [
                os.path.join(output_dir, f"result_{i}.jpg") 
                for i in range(len(results))
            ]
        
        return processed_paths, 1.0
    
    def download_results(self) -> str:
        """Prepare results for download."""
        # Create a zip file of results
        import shutil
        import zipfile
        
        zip_path = os.path.join(self.temp_dir, "results.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.temp_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.tif', '.geojson')):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.temp_dir)
                        zipf.write(file_path, arcname)
        
        return zip_path

def run_web_ui(model_paths: Dict[str, str], 
              host: str = "0.0.0.0", 
              port: int = 7860,
              share: bool = False):
    """
    Run the Gradio web interface.
    
    Args:
        model_paths: Dictionary of model paths
        host: Host to run the server on
        port: Port to run the server on
        share: Whether to create a public link
    """
    # Initialize detector
    detector = RemoteSensingDetector(model_paths)
    
    # Create and launch interface
    web_ui = RemoteSensingWebUI(detector)
    interface = web_ui.create_interface()
    
    print(f"Starting web interface at http://{host}:{port}")
    interface.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True
    )
