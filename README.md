# 🌍 Remote Sensing Urban Planner

A powerful tool for detecting and segmenting buildings and trees in aerial/satellite imagery using YOLOv8. This project provides both a Python API and a web interface for easy interaction.

## 🚀 Features

- **Multi-Model Detection**: Combines multiple YOLOv8 models for comprehensive analysis
- **Web Interface**: Interactive UI for easy model interaction
- **Batch Processing**: Process multiple images or entire directories at once
- **Customizable**: Adjust confidence thresholds and other parameters
- **Export Results**: Save detections in multiple formats

## 🛠️ Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- pip package manager

## 🚀 Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Remote_sensing.git
cd Remote_sensing

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
Remote_sensing/
├── app.py                # Main application entry point
├── setup.py              # Package configuration
├── requirements.txt      # Project dependencies
├── data/                 # Directory for input/output data
│   ├── raw/              # Raw input images
│   └── processed/        # Processed results
├── models/               # Directory for model weights
├── src/                  # Source code
│   ├── __init__.py       # Package initialization
│   ├── detector.py       # Core detection logic
│   └── web_interface.py  # Web UI implementation
└── README.md             # This file
```

## 🖥️ Web Interface

Launch the interactive web interface:

```bash
python app.py --web
```

Access the interface at `http://localhost:7860`

## 💻 Command Line Usage

### Process a single image
```bash
python app.py --source path/to/image.jpg --output outputs/
```

### Process a directory of images
```bash
python app.py --source path/to/images/ --output outputs/ --batch
```

### Available Arguments
```
--source       Path to input image, video, or directory
--output       Output directory (default: 'outputs/')
--conf         Confidence threshold (default: 0.25)
--device       Device to run on ('cuda:0', 'cpu', etc.)
--web          Launch web interface
--host         Web server host (default: '0.0.0.0')
--port         Web server port (default: 7860)
--share        Create a public link for the web interface
```

## 🏗️ Model Architecture

The system uses a combination of YOLOv8 models:

1. **Building Detection**
   - Custom YOLOv8 model
   - Trained on aerial imagery
   - Outputs bounding boxes around buildings

2. **Tree Segmentation**
   - Fine-tuned YOLOv8 segmentation model
   - Precisely segments tree canopies
   - Outputs pixel-level masks

3. **Building Segmentation**
   - Pre-trained model from Hugging Face
   - Segments building footprints
   - Provides detailed outlines of structures

## 📊 Training

To train the models on your own dataset:

1. Prepare your dataset in YOLO format
2. Update the configuration files
3. Run the training script:

```bash
python train.py --data data.yaml --cfg yolov8m.yaml --weights yolov8m.pt --batch 16 --epochs 50
```

### Training Parameters
- Image size: 640x640 pixels
- Batch size: 16 (adjust based on GPU memory)
- Learning rate: 0.0005
- Augmentation: Enabled
- Early stopping: 20 epochs patience

## 📦 Model Zoo

Pre-trained models are available for download:

| Model | Type | Dataset | mAP@0.5 | Download |
|-------|------|---------|---------|----------|
| YOLOv8m | Building Detection | Custom | 0.78 | [Download](https://example.com/models/building_det.pt) |
| YOLOv8m | Tree Segmentation | Custom | 0.72 | [Download](https://example.com/models/tree_seg.pt) |
| YOLOv8m | Building Segmentation | COCO | 0.85 | [Hugging Face](https://huggingface.co/keremberke/yolov8m-building-segmentation) |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for dataset management
- [Hugging Face](https://huggingface.co/) for model hosting
    mosaic=1.0,
    augment=True,
    name="yolov8m_house_results",
    )

# model_tree.train(
    data=f"{dataset_location_tree}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=5,
    cache="disk",
    device=0,
    workers=4,
    pretrained=True,
    lr0=0.0005,
    mosaic=1.0,
    augment=True,
    name="yolov8m_tree_results",
    )
# Validate Models
* After training, validate the models:
  * model_house.val()
  * model_tree.val()
# Training Outputs
* Models are saved in runs/detect/yolov8m_house_results/weights/best.pt and runs/detect/yolov8m_tree_results/weights/best.pt.
* Training metrics are logged to TensorBoard (see "Plot Metrics" section).
* Usage
# Inference
* Upload an Image:
  * The script prompts you to upload an image via Google Colab’s file upload feature:
  * from google.colab import files
  * uploaded = files.upload()
  * img_path = list(uploaded.keys())[0]
  * For local use, modify this to specify the image path directly:
  * upload a TIF files using Tools such as QGIS
  * img_path = "path/to/your/image.jpg"
#  Load Models
* House Detection Model:
  * model_house = YOLO("runs/detect/yolov8m_house_results/weights/best.pt")
# Building Segmentation Model (from Hugging Face):
* model_path = hf_hub_download(repo_id="keremberke/yolov8m-building-segmentation", filename="best.pt")
* model_building = YOLO(model_path)
# Tree Segmentation Model (currently commented out):
* model_tree = YOLO("runs/detect/yolov8m_tree_results/weights/best.pt")
# Run Inference
* The script processes the image with all models and combines the results:
  * Houses: Blue bounding boxes with confidence scores.
  * Buildings: Red overlay for segmented areas.
  * Trees: Green overlay (if uncommented).
* Example:
  * results_house = model_house(image_np)
  * results_building = model_building(image_np)
# Save and Display Results
* The combined output is saved as <original_filename>_combined_output.jpg in /content/output.
* The result is displayed using Matplotlib and downloaded via Colab’s files.download().
* Example Command (Local Environment)
* If running locally, modify the script to remove Colab-specific imports (google.colab) and run:
  * python anirudh_tree_and_building_det.py
# Input and Output Examples
* Input Image: An image containing houses, trees, and buildings (e.g., Input Image on Google Drive).
* Output Image: The processed image with:
  * Blue bounding boxes for houses.
  * Red overlays for buildings.
  * (Optional) Green overlays for trees.
* Example: Output Image on Google Drive.
  * Plot Metrics
# Start TensorBoard
Training metrics are saved to TensorBoard logs in the runs directory. To visualize them:
* In Colab, use:
  * %load_ext tensorboard
  * %tensorboard --logdir runs
* Locally, run:
  * tensorboard --logdir runs
  * Then open the provided URL (e.g., http://localhost:6006) in your browser.
# Plot Results Manually
* If training is interrupted, you can plot partial results:
  * from ultralytics.utils import plot_results
  * plot_results()
# Results
* House Detection: Achieves accurate bounding box detection for houses, with confidence scores displayed.
* Building Segmentation: Segments buildings effectively using the pre-trained model from Hugging Face.
* Tree Segmentation: (If enabled) Segments trees with a green overlay, though this is currently commented out.
# Performance
* The models are trained with a medium YOLOv8 architecture, balancing accuracy and speed.
* Training on a Tesla T4 GPU with the specified parameters takes approximately 20 epochs to converge (adjustable).
* Inference is fast, taking a few seconds per image on a GPU.
* Troubleshooting
# CUDA Errors
* Ensure CUDA is installed and compatible with your PyTorch version.
* Check GPU availability:
* import torch
*  print(torch.cuda.is_available())
# Roboflow API Key
* Replace the API key in the script with your own from Roboflow.
* Model Weights Not Found:
  * Train the models first or ensure the paths to best.pt files are correct.
# Colab-Specific Code
* Remove google.colab imports and modify file upload/download logic for local use.
# Future Improvements
* Enable Tree Segmentation: Uncomment the tree segmentation inference code and ensure the model weights are available.
* Hyperparameter Tuning: Adjust training parameters (e.g., epochs, lr0, imgsz) for better accuracy.
* Multi-Class Support: Extend the models to detect/segment additional classes (e.g., roads, vehicles).
* Local Deployment: Add a command-line interface for easier local usage.
* Model Optimization: Use quantization or pruning to reduce model size and improve inference speed.
# Contributing
* Contributions are welcome! To contribute:
  * Fork the repository.
  * Create a new branch (git checkout -b feature/your-feature).
  * Commit your changes (git commit -m "Add your feature").
  * Push to the branch (git push origin feature/your-feature).
  * Open a Pull Request.
# License
* This project is licensed under the MIT License. See the LICENSE file for details.
# Acknowledgments
* Ultralytics: For the YOLOv8 framework (GitHub).
* Roboflow: For hosting the datasets (house_alloc and tree-seg).
* Hugging Face: For the pre-trained building segmentation model (keremberke/yolov8m-building-segmentation).
* Google Colab: For providing the environment to develop and test the project.
# Contact
* For questions or issues, please open an issue on GitHub or contact the repository owner at <anirudh21india@gmail.com>
