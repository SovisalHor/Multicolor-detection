import os
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---- CONFIG ----
pred_json = "runs/detect/val2/predictions.json"
labels_dir = "D:/MultiColor.v2-mcolor.yolov8/valid/labels"
class_names = ["bluecube", "green cube", "red cube"]
# ----------------

# Load predictions
with open(pred_json, "r") as f:
    preds = json.load(f)

# Build prediction dictionary
pred_dict = {}
for p in preds:
    img_id = p["image_id"]            # exact name
    pred_class = p["category_id"]

    if img_id not in pred_dict:
        pred_dict[img_id] = []
    pred_dict[img_id].append(pred_class)

true_labels = []
pred_labels = []

# Walk through label files
for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    img_id = file.replace(".txt", "")   # matches prediction key
    filepath = os.path.join(labels_dir, file)

    # Ground truth labels
    with open(filepath, "r") as f:
        lines = f.readlines()

    gt_classes = [int(line.split()[0]) for line in lines]

    predicted = pred_dict.get(img_id, [])

    if len(predicted) == 0:
        continue   # no predictions → skip

    # Match 1-to-1 by count
    min_len = min(len(gt_classes), len(predicted))

    for i in range(min_len):
        true_labels.append(gt_classes[i])
        pred_labels.append(predicted[i])

# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("YOLOv8 Confusion Matrix")
plt.show()
