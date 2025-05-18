"""
Main script to run the entire vessel detection training pipeline.
This script orchestrates:
1. Model download
2. Dataset verification
3. Model training
"""
import os
import sys
from pathlib import Path
import time

# Import custom modules
from download_model import download_yolov8s
from verify_dataset import verify_dataset
from yolo_train import train_yolo

def run_pipeline():
    """Run the complete training pipeline."""
    start_time = time.time()
    print("=" * 80)
    print("VESSEL DETECTION TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Download model if needed
    print("\n[Step 1/3] Downloading YOLOv8s model")
    model_path = download_yolov8s()
    if not model_path:
        print("Failed to download YOLOv8s model. Exiting.")
        return False
    
    # Step 2: Verify dataset
    print("\n[Step 2/3] Verifying dataset structure")
    if not verify_dataset():
        print("Dataset verification failed. Please check your dataset structure and data.yaml file.")
        return False
    
    # Step 3: Train model
    print("\n[Step 3/3] Training YOLOv8s model")
    try:
        results = train_yolo()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 80)
    print(f"TRAINING PIPELINE COMPLETED in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved in: {results.save_dir}")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    run_pipeline() 