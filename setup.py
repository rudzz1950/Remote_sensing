from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="remote-sensing-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for detecting buildings and trees in aerial/satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/remote-sensing-detector",
    packages=find_packages(),
    package_data={
        "": ["*.pt"],  # Include model files
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "ultralytics>=8.0.0",
        "gradio>=3.0.0",
        "streamlit>=1.10.0",
        "pandas>=1.3.0",
        "geopandas>=0.10.0",
        "shapely>=1.8.0",
        "rasterio>=1.2.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.3.0",
        "onnx>=1.10.0",
        "onnxruntime>=1.10.0",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.1",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=5.4.1",
        "pillow>=8.4.0",
    ],
    entry_points={
        'console_scripts': [
            'remote-sensing=app:main',
        ],
    },
)
