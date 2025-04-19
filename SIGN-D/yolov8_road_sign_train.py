# Road Sign Detection with YOLOv8
# Author: Cascade AI
#
# This script trains, evaluates, and analyzes a YOLOv8 model for road sign detection.
# Make sure your dataset is in YOLO format and update the DATA_YAML path accordingly.

# 1. Install and Import Required Libraries
try:
    import ultralytics
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'ultralytics', 'matplotlib', 'seaborn', 'scikit-learn', 'pandas'])

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 2. Set Paths and Parameters
# Path to your real dataset's data.yaml
data_yaml_path = 'data/data.yaml'  # Update if your data.yaml is elsewhere

MODEL_TYPE = 'yolov8n.pt'        # Or 'yolov8s.pt', etc.
EPOCHS = 50
IMG_SIZE = 640
PROJECT = 'runs/detect'
NAME = 'road_sign_train'

# 3. Visualize Dataset Distribution (Optional)
def plot_class_distribution(data_yaml):
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    train_labels_dir = os.path.join(os.path.dirname(data_yaml), 'labels/train')
    labels = []
    for label_file in os.listdir(train_labels_dir):
        with open(os.path.join(train_labels_dir, label_file)) as lf:
            for line in lf:
                class_id = int(line.split()[0])
                labels.append(class_names[class_id])
    plt.figure(figsize=(8, 4))
    sns.countplot(labels)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

# Uncomment to visualize class distribution
# plot_class_distribution(data_yaml_path)

# 4. Train the YOLOv8 Model
model = YOLO(MODEL_TYPE)
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=NAME,
    exist_ok=True
)

# 5. Evaluate the Model
val_results = model.val(
    data=data_yaml_path,
    imgsz=IMG_SIZE,
    project=PROJECT,
    name=NAME
)

# 6. Show Training Metrics
metrics = results.metrics
print("Metrics:", metrics)
if 'map50-95' in metrics:
    print(f"mAP50-95: {metrics['map50-95']:.4f}")
if 'precision' in metrics:
    print(f"Precision: {metrics['precision']:.4f}")
if 'recall' in metrics:
    print(f"Recall: {metrics['recall']:.4f}")

# 7. Plot Confusion Matrix (if available)
if hasattr(val_results, 'confusion_matrix'):
    cm = val_results.confusion_matrix
    class_names = model.names
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

# 8. Visualize Predictions
# test_imgs = ['path/to/test/image1.jpg', 'path/to/test/image2.jpg']  # Add your test images here
# for img_path in test_imgs:
#     results = model(img_path)
#     results[0].show()  # Show image with predictions

# 9. Save the Best Model File Path
best_model_path = os.path.join(PROJECT, NAME, 'weights', 'best.pt')
print(f"Best model saved at: {best_model_path}")

# 10. Additional Analysis: PR Curve, F1 Curve
print(f"Check {os.path.join(PROJECT, NAME)} for plots: PR curve, F1 curve, confusion matrix, etc.")

# 11. Inference Example
# img = 'path/to/your/test/image.jpg'
# pred = model.predict(source=img, save=True, imgsz=IMG_SIZE)
# print(f"Prediction results saved in {os.path.join(PROJECT, NAME, 'predict')}")
