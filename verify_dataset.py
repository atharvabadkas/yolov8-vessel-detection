"""
Script to verify the dataset structure and provide summary statistics.
"""
import os
import yaml
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

def verify_dataset():
    """Verifies the dataset structure and provides summary statistics."""
    # Load dataset configuration
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get paths
    train_path = Path(data_config['train'])
    val_path = Path(data_config['val'])
    classes = data_config['names']
    
    print(f"Dataset configuration loaded successfully")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {list(classes.values())}")
    
    # Check if directories exist
    if not train_path.exists():
        print(f"ERROR: Training directory not found at {train_path}")
        return False
    
    if not val_path.exists():
        print(f"ERROR: Validation directory not found at {val_path}")
        return False
    
    # Count images
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Total images: {len(train_images) + len(val_images)}")
    
    # Check for label files
    train_labels_path = train_path.parent / 'labels'
    val_labels_path = val_path.parent / 'labels'
    
    if not train_labels_path.exists():
        print(f"WARNING: Training labels directory not found at {train_labels_path}")
    
    if not val_labels_path.exists():
        print(f"WARNING: Validation labels directory not found at {val_labels_path}")
    
    # Count label files if they exist
    train_labels = list(train_labels_path.glob('*.txt')) if train_labels_path.exists() else []
    val_labels = list(val_labels_path.glob('*.txt')) if val_labels_path.exists() else []
    
    print(f"Training label files: {len(train_labels)}")
    print(f"Validation label files: {len(val_labels)}")
    
    # Check if all images have corresponding label files
    if train_labels_path.exists():
        missing_train_labels = []
        for img in train_images:
            label_file = train_labels_path / f"{img.stem}.txt"
            if not label_file.exists():
                missing_train_labels.append(img.name)
        
        if missing_train_labels:
            print(f"WARNING: {len(missing_train_labels)} training images missing label files")
            print(f"First 5 examples: {missing_train_labels[:5]}")
    
    if val_labels_path.exists():
        missing_val_labels = []
        for img in val_images:
            label_file = val_labels_path / f"{img.stem}.txt"
            if not label_file.exists():
                missing_val_labels.append(img.name)
        
        if missing_val_labels:
            print(f"WARNING: {len(missing_val_labels)} validation images missing label files")
            print(f"First 5 examples: {missing_val_labels[:5]}")
    
    print("Dataset verification completed")
    return True

if __name__ == "__main__":
    verify_dataset() 