# Point Cloud Registration Benchmark & Visualization Tool

A comprehensive framework and interactive graphical interface for evaluating 3D point cloud registration algorithms. This project provides both an automated testbench for rigorous quantitative analysis and an Open3D-based UI for visual inspection and manual alignment. It seamlessly integrates classical registration pipelines with state-of-the-art deep learning models.

## Features

* **Interactive GUI (`main.py`)**: 
  * Load source and target `.ply` files.
  * Adjust initial poses manually via sliders or apply random perturbations.
  * Introduce synthetic degradations (Gaussian noise, cropping/partial overlap).
  * Run algorithms and instantly visualize the alignment.
* **Automated Testbench (`testbench.py`)**: 
  * Batch process large datasets of point clouds.
  * Automatically evaluate robustness against varying levels of noise, sparsity, and partial overlap.
  * Export comprehensive metrics (Time, Fitness, Inlier RMSE, Translation/Rotation Error) to CSV.
* **Supported Algorithms**:
  * **Classical**: ICP (Point-to-Point & Point-to-Plane), RANSAC, Fast Global Registration (FGR).
  * **Deep Learning (AI)**: Deep Closest Point (DCP) and Partial Registration Network (PRNet).

## Prerequisites

* **Python:** 3.8+ (Built in Python 3.8.10)
* **Environment:** Linux is required because of the Learning 3D library. For further requirements, check their GitHub page. 
* **Hardware:** CUDA-compatible GPU recommended for AI model inference.

### Dependencies
Install the required standard packages using your package manager:
`pip install learning3d open3d numpy scipy torch torchvision`

*Note on AI Models:* To use DCP and PRNet, the `learning3d` library is required. Make sure it is installed and configured properly in your environment.

## Usage

### 1. Graphical User Interface
Launch the interactive visualizer and benchmark tool by running: `python main.py`

*Note: The application explicitly configures software rendering (`LIBGL_ALWAYS_SOFTWARE=1`). If you have capable hardware, please change this setting.*

### 2. Automated Testbench
To run the batch evaluation pipeline across a dataset of `.ply` models, execute: `python testbench.py`

* Before running, ensure your dataset is placed in the `pointClouds` directory (or update `DATASET_DIRECTORY` in the script).
* Results will be generated in the `output/` folder as a timestamped CSV.

### 3. Downsampling Utility
A standalone script is provided to quickly downsample dense `.ply` files. Run it using:
`python downsize_ply.py <path_to_ply_file> <ratio>`
*(Example: `python downsize_ply.py scan.ply 0.5`)*

## Project Structure

* `main.py` / `ui_window.py`: Open3D graphical interface and application logic.
* `testbench.py`: Automated benchmarking pipeline for generating quantitative results across various conditions.
* `registration_algos.py`: Core registration logic wrapping Open3D functions and AI models.
* `ai_wrappers.py`: PyTorch wrappers for initializing and running Learning3D models (DCP, PRNet).
* `pointcloud_manager.py`: State management for source and target point clouds within the GUI.
* `util.py`: Helper functions for error computation (Translation/Rotation error), dynamic voxel sizing, and matrix transformations.
* `downsize_ply.py`: CLI tool for rapid point cloud downsampling.
