import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import shutil
from sklearn.model_selection import train_test_split

def create_dataset_splits(source_dir, target_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split a dataset into training, validation, and test sets
    
    Args:
        source_dir: Directory containing class folders with images
        target_dir: Directory to save the split datasets
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
    """
    # Create target directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'validation')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get class folders
    class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    for class_folder in class_folders:
        # Create class directories in each split
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)
        
        # Get all images in the class folder
        class_path = os.path.join(source_dir, class_folder)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split into train+val and test
        train_val_files, test_files = train_test_split(
            images, test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and val
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Copy files to respective directories
        for file in train_files:
            shutil.copy(
                os.path.join(class_path, file),
                os.path.join(train_dir, class_folder, file)
            )
        
        for file in val_files:
            shutil.copy(
                os.path.join(class_path, file),
                os.path.join(val_dir, class_folder, file)
            )
        
        for file in test_files:
            shutil.copy(
                os.path.join(class_path, file),
                os.path.join(test_dir, class_folder, file)
            )
        
        print(f"Class {class_folder}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")

def visualize_samples(data_dir, num_samples=5, figsize=(15, 10)):
    """
    Visualize random samples from each class in the dataset
    
    Args:
        data_dir: Directory containing class folders with images
        num_samples: Number of samples to visualize per class
        figsize: Figure size for the plot
    """
    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    num_classes = len(class_folders)
    
    plt.figure(figsize=figsize)
    
    for i, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select random samples
        samples = random.sample(images, min(num_samples, len(images)))
        
        for j, sample in enumerate(samples):
            img_path = os.path.join(class_path, sample)
            img = Image.open(img_path)
            
            plt.subplot(num_classes, num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.axis('off')
            
            if j == 0:
                plt.title(class_folder)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'sample_visualization.png'))
    plt.close()

def preprocess_image_file(image_path, target_size=(224, 224)):
    """
    Preprocess a single image file for model input
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = img.convert('RGB')  # Ensure 3 channels
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def analyze_dataset(data_dir):
    """
    Analyze a dataset and print statistics
    
    Args:
        data_dir: Directory containing class folders with images
        
    Returns:
        Dictionary with dataset statistics
    """
    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    stats = {
        'total_images': 0,
        'classes': {},
        'class_distribution': {}
    }
    
    for class_folder in class_folders:
        class_path = os.path.join(data_dir, class_folder)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        num_images = len(images)
        
        stats['classes'][class_folder] = num_images
        stats['total_images'] += num_images
    
    # Calculate class distribution
    for class_name, count in stats['classes'].items():
        stats['class_distribution'][class_name] = count / stats['total_images']
    
    # Print statistics
    print(f"Dataset Directory: {data_dir}")
    print(f"Total Images: {stats['total_images']}")
    print("Class Distribution:")
    for class_name, percentage in stats['class_distribution'].items():
        print(f"  - {class_name}: {percentage:.2%} ({stats['classes'][class_name]} images)")
    
    return stats