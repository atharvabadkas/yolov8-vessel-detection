from ultralytics import YOLO
import cv2
import yaml
import os
import torch
from pathlib import Path
import heapq
from typing import List, Tuple

def get_top_n_predictions(boxes, class_names: dict, n: int = 3) -> List[Tuple[str, float, int]]:
    """Get top N predictions with highest confidence scores."""
    predictions = []
    for box in boxes:
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        predictions.append((class_name, conf, class_id))
    
    # Sort by confidence and get top N
    return heapq.nlargest(n, predictions, key=lambda x: x[1])

def process_images(model_path: str, image_folder: str):
    """Process all images in a folder and show top 3 predictions for each."""
    # Set device (MPS for M1 Mac GPU, CPU as fallback)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load the model
    model = YOLO(model_path)
    model.to(device)  # Move model to MPS device
    
    # Load class names from data.yaml
    yaml_path = "data.yaml"
    with open(yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # Sort the image files to maintain consistent order
    image_files.sort()
    total_images = len(image_files)
    predicted_count = 0
    
    # Create output directories
    output_dir = "predicted_images"
    no_pred_dir = "no_predictions"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(no_pred_dir, exist_ok=True)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_file}")
            continue
        
        # Resize image while maintaining aspect ratio
        target_width = 640
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)
        resized_image = cv2.resize(image, (target_width, target_height))
        
        # Perform prediction
        results = model.predict(resized_image, conf=0.10, device=device, verbose=False)
        
        # Get top 3 predictions
        if len(results) > 0:
            boxes = results[0].boxes
            top_predictions = get_top_n_predictions(boxes, class_names)
            
            # Save annotated image with top prediction
            if top_predictions:
                predicted_count += 1
                
                # Print prediction information in terminal
                print("\n" + "="*50)
                print(f"Image name: {image_file}")
                for pred_class, pred_conf, _ in top_predictions:
                    print(f"Prediction: {pred_class}")
                    print(f"Accuracy: {pred_conf:.2%}")
                print("="*50)
                
                # Draw only the top prediction
                top_class, top_conf, top_class_id = top_predictions[0]
                
                # Find the box corresponding to the top prediction
                for box in boxes:
                    if int(box.cls[0].item()) == top_class_id:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cv2.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{top_class}: {top_conf:.2%}"
                        cv2.putText(resized_image, label, (int(x1), int(y2)+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
                
                output_path = os.path.join(output_dir, f"pred_{image_file}")
                cv2.imwrite(output_path, resized_image)
            else:
                # Save images with no predictions
                no_pred_path = os.path.join(no_pred_dir, f"no_pred_{image_file}")
                cv2.imwrite(no_pred_path, resized_image)
        else:
            # Save images with no predictions
            no_pred_path = os.path.join(no_pred_dir, f"no_pred_{image_file}")
            cv2.imwrite(no_pred_path, resized_image)

    print(f"Total images processed: {total_images}")
    print(f"Images with predictions: {predicted_count}")
    print(f"Images without predictions: {total_images - predicted_count}")
    print("\nCheck 'predicted_images' folder for images with detections")
    print("Check 'no_predictions' folder for images without detections")

if __name__ == "__main__":
    # Model path and image folder
    model_path = "vessel_detection/yolov8s_model/weights/best.pt"
    image_folder = "20250409"
    
    process_images(model_path, image_folder)
    print("Testing completed! Look at predicted images folder.")
