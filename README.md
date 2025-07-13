# 🍎 EatSmart AI - Food Image Classifier & Calorie Estimator

<div align="center">
  <img src="static/images/logo.png" alt="EatSmart AI Logo" width="200"/>
  
  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

## 🌟 Overview

EatSmart AI is an intelligent food recognition and calorie estimation system that uses deep learning to automatically classify food items from images and estimate their nutritional content. Built with a modern Flask web interface and powered by Convolutional Neural Networks (CNN), it helps users maintain healthy eating habits by providing instant calorie information.

### ✨ Key Features

- **🎯 Accurate Food Classification**: CNN-based model trained on diverse food datasets
- **📊 Real-time Calorie Estimation**: Instant nutritional information for classified foods
- **🎨 Modern Web Interface**: Beautiful, responsive design with smooth animations
- **📱 Mobile-Friendly**: Optimized for all devices and screen sizes
- **🔧 Easy Training**: Comprehensive training scripts for custom datasets
- **📈 Scalable Architecture**: Modular design for easy feature additions

## 🍽️ Supported Food Classes

The system currently supports classification of the following food items:

| Food Item | Calories (per 100g) | Food Item | Calories (per 100g) |
|-----------|-------------------|-----------|-------------------|
| 🍎 Apple | 52 kcal | 🍌 Banana | 89 kcal |
| 🫘 Beans | 337 kcal | 🥚 Boiled Egg | 155 kcal |
| 🍩 Doughnut | 452 kcal | 🍇 Grape | 62 kcal |
| 🥮 Mooncake | 266 kcal | 🍊 Orange | 47 kcal |
| 🍝 Pasta | 131 kcal | 🥝 Kiwi | 61 kcal |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Conda (recommended for environment management)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kamaljyotsingh/eatsmart-ai.git
   cd eatsmart-ai
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n eatsmart python=3.11
   conda activate eatsmart
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📸 Screenshots

### 🏠 Home Page
![Home Page](screenshots/Screenshot%202025-07-13%20at%203.19.45%E2%80%AFPM.png)

### 🔍 Prediction Interface
![Prediction](screenshots/Screenshot%202025-07-13%20at%203.19.55%E2%80%AFPM.png)

### 📊 Results Display
![Results](screenshots/Screenshot%202025-07-13%20at%203.20.13%E2%80%AFPM.png)

## 🧠 Model Architecture

### CNN Architecture
Our custom CNN model consists of:
- **Input Layer**: 224x224x3 RGB images
- **Convolutional Layers**: Multiple conv layers with ReLU activation
- **Pooling Layers**: Max pooling for dimension reduction
- **Dropout**: Regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Softmax activation for multi-class classification

### Training Process
1. **Data Preprocessing**: Image resizing, normalization, and augmentation
2. **Model Training**: Using TensorFlow/Keras with callbacks
3. **Validation**: Cross-validation for model evaluation
4. **Model Saving**: Best model saved as `.h5` file

## 🎯 Usage

### Web Interface
1. **Upload Image**: Click "Choose File" to select a food image
2. **Analyze**: Click "Analyze Image" to process the photo
3. **View Results**: See classification results and calorie estimates
4. **Get Details**: View nutritional information and serving suggestions

### Training Custom Model
See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions on:
- Dataset preparation
- Model training
- Hyperparameter tuning
- Model evaluation

## 📁 Project Structure

```
eatsmart-ai/
├── run.py                 # Main Flask application
├── model.py              # CNN model architecture
├── calorie.py            # Calorie estimation logic
├── segmentimage.py       # Image segmentation utilities
├── train_real_model.py   # Training script for real data
├── requirements.txt      # Python dependencies
├── static/               # Static assets
│   ├── images/          # Sample images and logos
│   └── styles/          # CSS stylesheets
├── templates/            # HTML templates
│   ├── index.html       # Home page
│   ├── predict.html     # Prediction page
│   └── layout.html      # Base template
├── images/              # Processing examples
└── model/               # Trained models (not in repo)
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configurations:
```env
FLASK_ENV=development
MODEL_PATH=model/food_classifier_model.h5
LABELS_PATH=label.npy
```

### Model Settings
Update `run.py` to use your trained model:
```python
MODEL_NAME = "your_trained_model.h5"  # Update this after training
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset Sources**: FOODD Dataset and ECUST Food Dataset
- **Deep Learning**: TensorFlow and Keras frameworks
- **Web Framework**: Flask for the web interface
- **UI Design**: Modern CSS with gradient animations

## 📞 Support

If you encounter any issues or have questions:
- 📧 Create an issue on GitHub
- 🐛 Report bugs with detailed descriptions
- 💡 Suggest new features

---

<div align="center">
  <p>Made with ❤️ for healthy eating</p>
  <p>⭐ Star this repository if you found it helpful!</p>
</div>





