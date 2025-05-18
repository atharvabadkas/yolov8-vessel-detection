"""
Script to download the YOLOv8s pre-trained model if not already available.
"""
from ultralytics import YOLO
import os
from pathlib import Path

def download_yolov8s():
    """Downloads the YOLOv8s model if not already present."""
    model_path = Path('yolov8s.pt')
    
    if model_path.exists():
        print(f"YOLOv8s model already exists at {model_path.absolute()}")
        # Validate the model
        try:
            model = YOLO(model_path)
            print("Model loaded successfully!")
            return model_path
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Will download a fresh copy.")
            # Continue with download if validation fails
    
    # Download the model
    print("Downloading YOLOv8s model...")
    try:
        model = YOLO('yolov8s.pt')  # This will download the model if not present
        print(f"YOLOv8s model downloaded successfully to {model_path.absolute()}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    download_yolov8s() 