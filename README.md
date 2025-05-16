# High Resolution Maps for Urban Planning
Overview
* Tree and BuildingDetector is a computer vision project that leverages the YOLOv8 framework to perform object detection and segmentation on images. Specifically, it:
  * Detects houses using bounding boxes.
  * Segments trees and buildings using masks.
  * Combines the results into a single output image with visual annotations.
* The project uses two custom-trained YOLOv8 models (one for house detection and one for tree segmentation) and a pre-trained YOLOv8 model from Hugging Face for building segmentation. It was originally developed in Google Colab and is designed to work with datasets from Roboflow.
* Key Features
  * House Detection: Identifies houses in images with bounding boxes (blue).
  * Tree Segmentation: Segments trees with a green overlay (currently commented out in the code).
  * Building Segmentation: Segments buildings with a red overlay using a pre-trained model from Hugging Face.
  * Combined Visualization: Outputs a single image with all detections and segmentations overlaid.
  * Custom YOLOv8 Models: Trains two YOLOv8 models (medium architecture) on custom datasets for houses and trees.
# Project Architecture
The project follows this workflow:
* Dataset Preparation: Downloads two datasets from Roboflow:
  * house_alloc (version 17) for house detection.
  * tree-seg (version 1) for tree segmentation.
  * Model Configuration: Defines custom YOLOv8 configurations for house detection and tree segmentation.
  * Training: Trains two YOLOv8 models on the respective datasets (training code is currently commented out).
# Inference
* Uses the trained house detection model.
* Uses a pre-trained building segmentation model from Hugging Face.
* (Optionally) Uses the trained tree segmentation model (commented out).
* Visualization: Combines results into a single annotated image and saves/downloads it.
* Metrics: Includes options to plot training metrics using TensorBoard (commented out).
# Prerequisites
* Python: Version 3.6 or higher.
* Hardware: A GPU (e.g., NVIDIA Tesla T4) is recommended for training and inference. CPU can be used but will be slower.
* Environment: The script was developed in Google Colab, but it can be adapted for local environments.
# Dependencies
* The following Python packages are required:
  1. ultralytics (for YOLOv8)
  2. torch (PyTorch)
  3. torchvision
  4. numpy
  5. opencv-python (cv2)
  6. matplotlib
  7. grad-cam
  8. roboflow (for dataset download)
  9. huggingface_hub (for downloading the building segmentation model)
  10. pillow (PIL for image processing)
Install them using:
* pip install ultralytics torch torchvision numpy opencv-python matplotlib grad-cam roboflow huggingface_hub pillow
# Setup Instructions
Clone the Repository:
* git clone https://github.com/<your-username>/TreeAndBuildingDetector.git
* cd TreeAndBuildingDetector
* Set Up a Virtual Environment (Optional but Recommended):
  * python -m venv venv
  * source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install Dependencies
* pip install -r requirements.txt
* If you don’t have a requirements.txt file yet, create one with the dependencies listed above:
  1. ultralytics
  2. torch
  3. torchvision
  4. numpy
  5. opencv-python
  6. matplotlib
  7. grad-cam
  8. roboflow
  9.huggingface_hub
  10. pillow
* Then run the install command.
# Download the YOLOv8 Repository
* The script uses the Ultralytics YOLOv8 framework. Clone it:
  * git clone https://github.com/ultralytics/ultralytics.git
# Prepare Datasets
* The script downloads two datasets from Roboflow using an API key:
  * House Dataset: house_alloc (version 17) from the quantela workspace.
  * Tree Dataset: tree-seg (version 1) from the test-4udyq workspace.
  * To download these datasets, you need a Roboflow API key. Replace "iMVOMaxCVf9Q6wQNbSnb" in the script with your own API key:
  * rf = Roboflow(api_key="YOUR_API_KEY")
  * Alternatively, you can manually download the datasets from Roboflow and place them in the project directory (e.g., /content/house_alloc-17 and /content/tree-seg-1 in Colab).
# Download Pre-trained Models (Optional)
* If you’re not training the models, you’ll need the trained weights
  * House detection model: runs/detect/yolov8m_house_results/weights/best.pt
  * Tree segmentation model: runs/detect/yolov8m_tree_results/weights/best.pt (currently commented out)
  * Building segmentation model: Automatically downloaded from Hugging Face (keremberke/yolov8m-building-segmentation).
  * If these files are not available, you’ll need to train the models (see the "Training" section).
Datasets
# House Allocation Dataset (house_alloc)
* Source: Roboflow (quantela workspace, version 17).
* Classes: 1 (Houses).
* Structure: Contains train, valid, and test splits with images and labels in YOLO format.
* Location: /content/house_alloc-17 (in Colab).
# Tree Segmentation Dataset (tree-seg)
* Source: Roboflow (test-4udyq workspace, version 1).
* Classes: 1 (assumed Trees, verify in data.yaml).
* Structure: Contains train, valid, and test splits with images and labels in YOLO format.
* Location: /content/tree-seg-1 (in Colab).
# Building Segmentation Model
* Source: Hugging Face (keremberke/yolov8m-building-segmentation).
* Pre-trained model for building segmentation.
* Dataset Configuration
* The script updates the data.yaml files for both datasets to ensure correct paths for train, valid, and test splits:
  * train: train/images
  * val: valid/images
  * test: test/images
# Model Configuration
The script defines two custom YOLOv8 configurations:
  * House Detection Model (yolov8m-custom-house.yaml)
  * Based on YOLOv8 medium architecture.
  * Number of classes: 1 (Houses).
  * Uses a detection head (Detect).
  * Tree Segmentation Model (yolov8m-custom-tree.yaml)
  * Based on YOLOv8 medium architecture.
  * Number of classes: 1 (Trees).
  * Uses a segmentation head (Segment).
  * Both configurations adjust the depth and width multiples for a medium-scale model (depth_multiple: 0.67, width_multiple: 0.75).
# Training
* The script includes code to train both models (currently commented out). To train the models:
  * Uncomment the Training Section in anirudh_tree_and_building_det.py.
# Set Up GPU (if available)
* The script checks for CUDA availability and uses the GPU if present.
* Example output:
  * Setup complete. Using torch 2.3.0 on CUDA
    (True
    1
    Tesla T4)
# Training Parameters
* Epochs: 20 (can be increased for better results).
* Image Size: 640x640 pixels.
* Batch Size: 16 (suitable for Tesla T4 GPU).
* Patience: 20 for house model, 5 for tree model (early stopping).
* Learning Rate: 0.0005 (for stability).
* Augmentation: Enabled (mosaic=1.0, augment=True).
* Device: GPU (device=0).
# Run Training
# model_house.train(
    data=f"{dataset_location_house}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    patience=20,
    cache="disk",
    device=0,
    workers=4,
    pretrained=True,
    lr0=0.0005,
    mosaic=1.0,
    augment=True,
    name="yolov8m_house_results",
    )

# model_tree.train(
    data=f"{dataset_location_tree}/data.yaml",
    epochs=20,
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
* For questions or issues, please open an issue on GitHub or contact the repository owner at <your-email> (replace with your email).
