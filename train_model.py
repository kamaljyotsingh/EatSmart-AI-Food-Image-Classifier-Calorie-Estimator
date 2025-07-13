import tensorflow as tf
import numpy as np
import os
from model import create_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_dummy_data():
    """Create dummy training data for demonstration"""
    # Generate random images (400x400x3) for 10 classes
    num_samples = 100  # 10 samples per class
    num_classes = 10
    
    # Create dummy images
    X_train = np.random.rand(num_samples, 400, 400, 3).astype(np.float32)
    
    # Create dummy labels (one-hot encoded)
    y_train = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        class_idx = i % num_classes
        y_train[i, class_idx] = 1
    
    return X_train, y_train

def train_model():
    """Train a simple model for demonstration"""
    print("Creating model...")
    model = create_model()
    
    print("Generating dummy training data...")
    X_train, y_train = create_dummy_data()
    
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model...")
    # Train for a few epochs with dummy data
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    model_path = 'model/food_classifier_model.h5'
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Create labels file if it doesn't exist
    labels = ['apple', 'banana', 'beans', 'egg', 'doughnut', 'mooncake', 'pasta', 'grape', 'orange', 'qiwi']
    np.save('label.npy', labels)
    print("Labels saved to: label.npy")
    
    print("Training completed!")
    return model_path

if __name__ == "__main__":
    model_path = train_model()
    print(f"\nModel is ready! Update MODEL_NAME in run.py to: {os.path.basename(model_path)}") 