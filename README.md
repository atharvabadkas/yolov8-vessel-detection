# 🍽️ Vessel Detection in Kitchen Environments using YOLOv8

A production-ready vessel detection system built using YOLOv8s and transfer learning, optimized for multi-class detection in real-world commercial kitchen environments. The system supports robust data augmentation, validation, visual feedback, and confidence-aware prediction pipelines on both Apple M1 (via MPS) and CPU setups.

---

## 📌 Table of Contents

- 🔍 Overview
- 📁 Project Structure
- ⚙️ Training Pipeline
- 🔬 Inference Pipeline
- 📊 Model Performance
- 🧠 Design Decisions
- 🧰 Dependencies
- 📝 Conclusion

---

## 🔍 Overview

This project aims to create a robust, scalable, and efficient vessel detection pipeline using YOLOv8. It includes:
- End-to-end training and inference workflows
- Dataset validation
- Confidence-based filtering
- Multi-class visual performance metrics
- Mac M1 (MPS) and CPU compatibility

---

## ⚙️ Training Pipeline

The pipeline is initiated through `run_training.py`, which executes:

### ✅ 3. YOLOv8 Model Training (`train_yolo()` in yolo_train.py)

| Parameter        | Value         | Description                           |
|------------------|---------------|---------------------------------------|
| `epochs`         | 100           | Maximum epochs for training           |
| `imgsz`          | 320           | Image size (balanced for speed/acc)   |
| `batch`          | 16            | Batch size                            |
| `optimizer`      | Adam          | Optimizer for fast convergence        |
| `device`         | MPS/CPU       | Mac M1 GPU or fallback to CPU         |
| `patience`       | 15            | Early stopping tolerance              |
| `lr0`            | 0.001         | Initial learning rate                 |
| `lrf`            | 0.01          | Final learning rate factor            |
| `dropout`        | 0.2           | Dropout for regularization            |
| `weight_decay`   | 0.0005        | Penalizes large weights               |
| `mosaic`         | 1.0           | Advanced data augmentation            |
| `copy_paste`     | 0.5           | CutMix-style object blending          |
| `hsv_h/s/v`      | 0.015/0.7/0.4 | Color-based augmentations             |
| `degrees`        | 15.0          | Rotation                              |
| `translate`      | 0.2           | Random translation                    |
| `scale`          | 0.5           | Random scale                          |
| `fliplr`         | 0.5           | Horizontal flips                      |

---
## 🔬 Inference Pipeline

The `yolo_test.py` script handles model evaluation on new images using:
process_images(model_path, image_folder)

### 🔄 Inference Workflow
- Loads YOLO model and reads classes from data.yaml
- Scans folder for .jpg, .jpeg, .png images
- Runs predictions (confidence > 0.10)
- Annotates top-1 bounding box with label and probability

Outputs:
- predicted_images/: Predictions with bounding boxes
- no_predictions/: Images with no confident detection

### 🔍 Confidence-Based Top-N Predictions
get_top_n_predictions(preds, top_n=3)


---

## 📊 Model Performance

### 📈 Training Curves (`results.png`)

Includes YOLO-generated plots of:

- Training Loss: box, cls, dfl
- Validation Loss: box, cls, dfl
- Precision and Recall curves
- mAP@0.5 and mAP@0.5:0.95 performance trends

### 🔁 Confusion Matrices

Unnormalized:
![confusion_matrix](https://github.com/user-attachments/assets/b87f80db-e5c5-42ab-848e-ef22aaa42848)

Normalized:
![confusion_matrix_normalized](https://github.com/user-attachments/assets/f96a50b7-62b1-4947-b38c-12e931f5b252)


### 📈 Precision-Recall and F1 Curves

Recall-Confidence:
![R_curve](https://github.com/user-attachments/assets/97ebae6c-3b0d-4fdc-890d-41608dc20659)



Precision-Confidence:
![P_curve](https://github.com/user-attachments/assets/875e9df2-ebac-4cd3-b691-15ebe71b3b56)



F1-Confidence:
![F1_curve](https://github.com/user-attachments/assets/746405c2-b77a-4945-a7c3-60cdbd089108)



Precision-Recall:
![PR_curve](https://github.com/user-attachments/assets/766e0a75-0c3c-467c-ad08-a3656487f31c)




## 🧮 Class-Wise Results (`results.csv`)

| Class              | Precision | Recall | F1   | mAP@.5 |
|--------------------|-----------|--------|------|--------|
| aluminium paraat   | 0.871     | 0.86   | 0.87 | 0.871  |
| dosa vessel        | 0.954     | 1.00   | 0.97 | 0.954  |
| stew vessel        | 0.829     | 1.00   | 0.91 | 0.829  |
| sandwich counter   | 0.681     | 0.90   | 0.77 | 0.681  |
| steel tray         | 0.763     | 0.95   | 0.83 | 0.763  |
| **All Classes**    | **0.96**  | **0.91** | **0.86** | **0.918** |

## 🧠 Design Decisions

1. **Transfer Learning**  
   - Starting from `yolov8s.pt` improves convergence and prevents overfitting

2. **Early Stopping**  
   - Patience-based termination prevents resource waste and boosts generalization

3. **MPS Support (Mac M1 GPU)**  
   - Uses Apple’s Metal Performance Shaders when available, falls back to CPU otherwise

4. **Confidence Thresholding**  
   - Custom thresholding ensures bounding boxes meet precision requirements

5. **Data Augmentation**  
   - Uses mosaic, cut-paste, HSV, and geometric transformations to improve generalization

## ✅ Summary of Key Advances

| Category               | Improvements                                                                 |
|------------------------|------------------------------------------------------------------------------|
| ✅ Pipeline Automation | `run_training.py` integrates download → verify → train flow                  |
| ✅ Augmentation        | Rotation, mosaic, scale, HSV, flipping, cut-paste                            |
| ✅ Training Stability  | Adam optimizer, dropout, weight decay, patience-based early stopping         |
| ✅ Mac Support         | M1 acceleration via MPS, fallback to CPU                                     |
| ✅ Visual Feedback     | Precision/Recall curves, F1 score, mAP curves, confusion matrix              |
| ✅ Confidence Control  | Adjustable thresholding, top-N predictions, batch processing support         |

## 📌 Conclusion

This YOLOv8-powered pipeline demonstrates a state-of-the-art kitchen vessel detection system designed for both performance and deployability. With >91% mAP@0.5, comprehensive evaluation metrics, and hardware-aware execution paths, the system is suitable for:

- Smart kitchen monitoring
- Real-time restaurant analytics
- Deployment on embedded or cloud setups

The modular codebase and visualization-rich metrics offer the perfect foundation for scaling, experimentation, or integration with other AI workflows.


## 📁 Project Structure

```bash

├── run_training.py           # Orchestrates model download, dataset verification, training
├── yolo_train.py             # Custom training loop using Ultralytics YOLOv8
├── yolo_test.py              # Batch inference and annotated output generation
├── download_model.py         # Downloads pretrained YOLOv8s weights
├── verify_dataset.py         # Validates dataset structure and label format
├── train_val_split.py        # Automatically splits dataset into train/val sets
├── data.yaml                 # YOLOv8 dataset configuration
├── requirements.txt          # Project dependencies
├── results.csv               # Class-wise precision, recall, F1, and confidence thresholds
├── results.png               # YOLO training curves
├── confusion_matrix.png      # Class-wise prediction heatmap
├── confusion_matrix_normalized.png
├── R_curve.png               # Recall vs Confidence
├── P_curve.png               # Precision vs Confidence
├── F1_curve.png              # F1 vs Confidence
├── PR_curve.png              # Precision-Recall curves
├── labels.jpg                # Class label frequency and bounding box anchor visualization
├── labels_correlogram.jpg    # Heatmap of positional + dimension correlations
