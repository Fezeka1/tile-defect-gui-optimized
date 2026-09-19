# Tile Defect Inspector

An optimized autoencoder-based anomaly detection system for identifying surface defects on ceramic tiles.

The system uses a pretrained ResNet18 encoder and a convolutional decoder to reconstruct defect-free tile images. Defects are detected from reconstruction error, and the application provides a simple Streamlit interface for image inspection.

## Features

- Upload one or more tile images
- Automatic GOOD / DEFECTIVE classification
- Anomaly heatmap for visual inspection
- Adjustable detection sensitivity
- Result history for previously analysed images
- Optimized anomaly scoring using the final trained model

## Installation

Clone the repository:

```bash
git clone https://github.com/Fezeka1/tile-defect-gui-optimized.git
cd tile-defect-gui-optimized
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

The application loads the final optimized model and calibration data from the project directory.

## Re-evaluating the Model

The MVTec AD Tile dataset is not included in this repository.

To re-evaluate the trained model, download the MVTec AD Tile dataset and run:

```bash
python evaluate.py --data-root /path/to/tile --checkpoint optimization2_model.pth --out calibration.json
```

To verify that the model checkpoint loads correctly:

```bash
python test_model.py
```

## Tech Stack

- Python
- PyTorch
- torchvision
- scikit-learn
- Streamlit
- Pillow
- NumPy
- Matplotlib
