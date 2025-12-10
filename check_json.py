import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---- CONFIG ----
pred_json = "runs/detect/val2/predictions.json"
labels_dir = "D:/MultiColor.v2-mcolor.yolov8/valid/labels"   # <-- change if needed
class_names = ["bluecube", "green cube", "red cube"]        # <-- adjust to your classes
# ----------------

# Load predictions
with open(pred_json, "r") as f:
    preds = json.load(f)

# Build dict of predictions:  image → list of predicted classes
pred_dict = {}
for p in preds:
    img_id = os.path.splitext(p["image_id"])[0]  # remove .jpg.rf.*.png stuff
    pred_class = p["category_id"]
    
    if img_id not in pred_dict:
        pred_dict[img_id] = []
    pred_dict[img_id].append(pred_class)

true_labels = []
pred_labels = []

# Walk through label files (GT)
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    img_id = label_file.replace(".txt", "")  # matches predictions key
    filepath = os.path.join(labels_dir, label_file)

    # Read ground-truth classes
    with open(filepath, "r") as f:
        lines = f.readlines()

    gt_classes = [int(line.split()[0]) for line in lines]

    # Predictions for this image
    predicted = pred_dict.get(img_id, [])

    # Match count between GT and predictions
    min_len = min(len(gt_classes), len(predicted))

    for i in range(min_len):
        true_labels.append(gt_classes[i])
        pred_labels.append(predicted[i])

# Convert to confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("YOLOv8 Confusion Matrix")
plt.show()
