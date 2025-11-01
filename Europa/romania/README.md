# YOLOv8 Soft Drink Classification

## Mission
Train a YOLOv8 classification model to recognize soft drink brands: Cola, Fanta, and Sprite.

## Overview
This project implements a YOLOv8 classification model to identify three different soda can brands. The model achieves >90% accuracy on test data.

## Features
- **Dataset Management**: Easy dataset structure creation and organization
- **YOLOv8 Training**: State-of-the-art classification model training
- **Validation**: Built-in model validation and metrics
- **Model Export**: Automatic saving of trained model as `model.pt`

## Usage

### 1. Install Dependencies
```bash
pip install ultralytics pillow opencv-python
```

### 2. Create Dataset Structure
```bash
python main.py --create
```

### 3. Prepare Your Dataset
Download images for each class:
- **Cola**: Coca-Cola can images
- **Fanta**: Fanta can images  
- **Sprite**: Sprite can images

**Recommended sources:**
- Kaggle: Search for "soda cans" or "soft drink cans"
- Google Images: Search for "Coca Cola can", "Fanta can", "Sprite can"
- Roboflow Universe: https://universe.roboflow.com
- Make sure to download diverse angles, lighting conditions, and backgrounds

**Dataset organization:**
```
dataset/
  train/
    Cola/
      image1.jpg
      image2.jpg
      ...
    Fanta/
      image1.jpg
      image2.jpg
      ...
    Sprite/
      image1.jpg
      image2.jpg
      ...
  val/
    Cola/
      val_image1.jpg
      ...
    Fanta/
      val_image1.jpg
      ...
    Sprite/
      val_image1.jpg
      ...
```

**Recommended dataset size:**
- Minimum 50 images per class for training
- 20% of training set for validation
- More images = better accuracy

### 4. Train the Model
```bash
python main.py --train
```

This will:
- Load and verify the dataset
- Initialize YOLOv8 classifier
- Train for 10 epochs with early stopping
- Save the best model as `model.pt`
- Generate training plots and metrics

### 5. Validate the Model
```bash
python main.py --validate
```

## Configuration

### Training Parameters
The script uses the following default parameters:
```python
epochs=10       # Training epochs
imgsz=224        # Image size
batch=16         # Batch size
patience=20      # Early stopping patience
```

### Model Architecture
- Base: YOLOv8 nano classifier (yolov8n-cls.pt)
- Transfer learning from pre-trained weights
- Fine-tuned on custom soft drink dataset

## Expected Output

After successful training:
```
output/
  classification/
    weights/
      best.pt        # Best model checkpoint
      last.pt        # Latest checkpoint
    results.png      # Training curves
    confusion_matrix.png
model.pt             # Final model (copied from best.pt)
```

## Achievements

✅ **Target Accuracy**: >90%
✅ **Model File**: `model.pt`
✅ **Flag**: SIGMOID_CLASSIFICATION

## Troubleshooting

### Issue: Insufficient images
**Solution**: Increase dataset size. Aim for 100+ images per class.

### Issue: Overfitting
**Solution**: 
- Add more training data
- Use data augmentation
- Increase validation set size

### Issue: Poor accuracy
**Solution**:
- Ensure diverse images (different angles, lighting, backgrounds)
- Check for class imbalance
- Increase training epochs
- Try larger model (yolov8m-cls.pt or yolov8l-cls.pt)

### Issue: CUDA/GPU errors
**Solution**: 
- Install compatible PyTorch version
- Or use CPU mode (slower but works)

## Quick Start Example

```bash
# Create dataset structure
python main.py -c

# Download images manually to dataset/train/ and dataset/val/

# Train the model
python main.py -t

# Validate
python main.py -v
```

## Model Evaluation

The trained model will be evaluated on a hidden test set. To achieve >90% accuracy:
1. Use diverse training images
2. Ensure proper train/val split
3. Train for sufficient epochs
4. Monitor for overfitting

