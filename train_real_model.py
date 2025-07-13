import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_images_from_folders(data_dir, img_size=(400, 400)):
    """
    Load images from folders where each folder is a class name
    data_dir structure should be:
    data_dir/
        apple/
            img1.jpg
            img2.jpg
        banana/
            img1.jpg
            img2.jpg
        ...
    """
    images = []
    labels = []
    class_names = []
    
    print("Loading images from folders...")
    
    # Get all class folders
    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    class_folders.sort()  # Sort for consistent ordering
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_name)
        class_names.append(class_name)
        
        print(f"Loading {class_name} images...")
        
        # Get all image files in this class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, img_size)
                    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
                    
                    images.append(img)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} images from {len(class_names)} classes")
    return np.array(images), np.array(labels), class_names

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators with augmentation for training
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=len(np.unique(y_train)))
    y_val_cat = to_categorical(y_val, num_classes=len(np.unique(y_train)))
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val_cat,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def train_model_with_real_data(data_dir, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train the model with real food images
    """
    print("=== Starting Model Training ===")
    
    # Load images and labels
    images, labels, class_names = load_images_from_folders(data_dir)
    
    if len(images) == 0:
        print("No images found! Please check your data directory structure.")
        return None
    
    print(f"Class names: {class_names}")
    print(f"Number of images per class:")
    for i, class_name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"  {class_name}: {count}")
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, 
        test_size=validation_split, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Create data generators
    train_generator, val_generator = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    
    # Update the last layer to match number of classes
    num_classes = len(class_names)
    model.layers[-1] = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    os.makedirs('model', exist_ok=True)
    final_model_path = 'model/food_classifier_final.h5'
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save class names
    np.save('label.npy', class_names)
    print(f"Class names saved to: label.npy")
    
    # Plot training history
    try:
        plot_training_history(history)
    except:
        print("Could not plot training history (matplotlib not available)")
    
    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n=== Training Complete ===")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Model saved to: {final_model_path}")
    print(f"Update MODEL_NAME in run.py to: food_classifier_final.h5")
    
    return model, class_names

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "dataset"  # Change this to your dataset path
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Check if dataset directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' not found!")
        print("Please create a dataset directory with the following structure:")
        print("dataset/")
        print("  apple/")
        print("    img1.jpg")
        print("    img2.jpg")
        print("  banana/")
        print("    img1.jpg")
        print("    ...")
        print("\nYou can download the datasets mentioned in the README:")
        print("- FOODD Dataset")
        print("- ECUST Food Dataset (ECUSTFD)")
    else:
        # Train the model
        model, class_names = train_model_with_real_data(
            DATA_DIR, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE
        ) 