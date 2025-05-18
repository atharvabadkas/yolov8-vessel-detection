from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolo():
    """
    Trains a YOLOv8s model on vessel dataset with:
    - Transfer learning from a pre-trained model
    - Data augmentation to enhance training data
    - Careful hyperparameter tuning
    - Proper model architecture configuration
    - Regularization to prevent overfitting
    - MPS acceleration on M1 Mac
    """
    # Set device (MPS for M1 Mac GPU, CPU as fallback)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create results directory
    save_dir = Path("runs/train")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize YOLOv8s model (transfer learning from pre-trained weights)
    model = YOLO('yolov8s.pt')  # Using YOLOv8s as requested
    
    # Configure training parameters
    results = model.train(
        data='data.yaml',              # Dataset config
        epochs=100,                    # Maximum number of epochs
        imgsz=320,                     # Image size for training (increased from 320)
        device=device,                 # Utilize MPS acceleration on M1
        patience=15,                   # Early stopping patience
        batch=16,                      # Batch size optimized for M1 Mac
        optimizer='Adam',              # Using Adam optimizer for faster convergence
        lr0=0.001,                     # Initial learning rate for fine-tuning
        lrf=0.01,                      # Final learning rate factor
        weight_decay=0.0005,           # Weight decay for regularization
        dropout=0.2,                   # Dropout for regularization
        mosaic=1.0,                    # Enable mosaic augmentation
        copy_paste=0.5,                # Copy-paste augmentation
        degrees=15.0,                  # Random rotation up to 15 degrees
        translate=0.2,                 # Random translation up to 20%
        scale=0.5,                     # Random scaling
        fliplr=0.5,                    # Horizontal flip probability
        hsv_h=0.015,                   # HSV hue augmentation
        hsv_s=0.7,                     # HSV saturation augmentation
        hsv_v=0.4,                     # HSV value augmentation
        save=True,                     # Save checkpoints
        plots=True,                    # Generate training plots
        project="vessel_detection",    # Project name
        name="yolov8s_model",          # Run name
        exist_ok=True                  # Overwrite existing run
    )

    print(f"Training completed! Results saved in: {results.save_dir}")
    
    # Report final metrics
    print(f"Final mAP50: {results.metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"Final mAP50-95: {results.metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    
    return results

if __name__ == "__main__":
    train_yolo() 