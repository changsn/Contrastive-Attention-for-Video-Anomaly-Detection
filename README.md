# Contrastive Attention for Video Anomaly Detection

## Data Preparation
1. Download UCF-Crime I3D pre-trained features from google drive https://drive.google.com/drive/folders/1O0lXlEFQ8OoMTK11yb2HxPP4kGVVR02o?usp=sharing
2. Place the feature files in folder data.

## Installation

**Requirements**
- Python >= 3.5
- [Pytorch](https://pytorch.org/) >= 0.4.0
- [matplotlib](https://matplotlib.org/)
- Numpy
- Pickle
- Scipy
- Sklearn

## Usage
Training:
```shell
python train.py
```

Inference:
```shell
python test.py
```
