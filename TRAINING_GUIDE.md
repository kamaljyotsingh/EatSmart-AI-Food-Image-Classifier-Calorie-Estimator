# 🍎 EatSmart AI - Model Training Guide

## 📋 Overview
This guide will help you train the EatSmart AI model with real food images to achieve accurate calorie estimation.

## 🗂️ Dataset Preparation

### Option 1: Download Existing Datasets
The original project mentions these datasets:
- **FOODD Dataset** - Food image dataset
- **ECUST Food Dataset (ECUSTFD)** - [https://github.com/Liang-yc/ECUSTFD-resized-](https://github.com/Liang-yc/ECUSTFD-resized-)

### Option 2: Create Your Own Dataset
Create a `dataset` folder with the following structure:
```
dataset/
├── apple/
│   ├── apple1.jpg
│   ├── apple2.jpg
│   └── ...
├── banana/
│   ├── banana1.jpg
│   ├── banana2.jpg
│   └── ...
├── beans/
│   ├── beans1.jpg
│   ├── beans2.jpg
│   └── ...
├── egg/
│   ├── egg1.jpg
│   ├── egg2.jpg
│   └── ...
├── doughnut/
│   ├── doughnut1.jpg
│   ├── doughnut2.jpg
│   └── ...
├── mooncake/
│   ├── mooncake1.jpg
│   ├── mooncake2.jpg
│   └── ...
├── pasta/
│   ├── pasta1.jpg
│   ├── pasta2.jpg
│   └── ...
├── grape/
│   ├── grape1.jpg
│   ├── grape2.jpg
│   └── ...
├── orange/
│   ├── orange1.jpg
│   ├── orange2.jpg
│   └── ...
└── qiwi/
    ├── qiwi1.jpg
    ├── qiwi2.jpg
    └── ...
```

### Supported Image Formats
- JPG/JPEG
- PNG
- GIF
- BMP

## 🚀 Training Steps

### 1. Prepare Your Dataset
```bash
# Create dataset directory
mkdir dataset

# Create class folders
mkdir dataset/apple dataset/banana dataset/beans dataset/egg
mkdir dataset/doughnut dataset/mooncake dataset/pasta dataset/grape
mkdir dataset/orange dataset/qiwi

# Add your images to each folder
# Example: Copy apple images to dataset/apple/
```

### 2. Run Training
```bash
# Activate your environment
conda activate TF-M1

# Run training script
python train_real_model.py
```

### 3. Training Configuration
You can modify these parameters in `train_real_model.py`:
- `DATA_DIR = "dataset"` - Path to your dataset
- `EPOCHS = 50` - Number of training epochs
- `BATCH_SIZE = 32` - Batch size for training

## 📊 Training Features

### ✅ What the Training Script Does:
- **Automatic Data Loading**: Loads images from folder structure
- **Data Augmentation**: Rotations, flips, zooms for better generalization
- **Train/Validation Split**: 80% training, 20% validation
- **Early Stopping**: Stops training if no improvement
- **Learning Rate Scheduling**: Reduces learning rate when stuck
- **Model Checkpointing**: Saves best model during training
- **Training Visualization**: Plots accuracy and loss curves

### 📈 Expected Training Time:
- **Small dataset** (100-500 images per class): 10-30 minutes
- **Medium dataset** (500-1000 images per class): 30-60 minutes
- **Large dataset** (1000+ images per class): 1-3 hours

## 🎯 After Training

### 1. Update Model Name
After training completes, update `run.py`:
```python
MODEL_NAME = 'food_classifier_final.h5'  # or 'best_model.h5'
```

### 2. Test Your Model
```bash
# Restart the Flask app
python run.py
```

### 3. Upload Test Images
- Go to http://127.0.0.1:5000/predict
- Upload food images to test your trained model

## 📁 Generated Files

After training, you'll get:
- `model/food_classifier_final.h5` - Final trained model
- `model/best_model.h5` - Best model during training
- `label.npy` - Updated class labels
- `training_history.png` - Training progress plots

## 🔧 Troubleshooting

### Common Issues:

1. **"Dataset directory not found"**
   - Create the `dataset` folder with proper structure

2. **"No images found"**
   - Check image file extensions (.jpg, .png, etc.)
   - Ensure images are in the correct class folders

3. **"Out of memory"**
   - Reduce `BATCH_SIZE` in the script
   - Use fewer images per class

4. **"Training too slow"**
   - Reduce `EPOCHS`
   - Use GPU if available
   - Reduce image resolution

## 📚 Tips for Better Results

1. **Image Quality**: Use high-quality, clear food images
2. **Variety**: Include different angles, lighting, and backgrounds
3. **Balance**: Have similar number of images per class
4. **Preprocessing**: Images are automatically resized to 400x400
5. **Augmentation**: The script automatically augments training data

## 🎉 Success Metrics

Good model performance indicators:
- **Training Accuracy**: > 85%
- **Validation Accuracy**: > 80%
- **No overfitting**: Validation accuracy close to training accuracy

## 📞 Need Help?

If you encounter issues:
1. Check the error messages carefully
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Try with a smaller dataset first

---

**Happy Training! 🍕🥗🍎** 