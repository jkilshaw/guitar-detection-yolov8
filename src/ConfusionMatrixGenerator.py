from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np

# --- paths ---
model_path = "/Users/jackkilshaw/PythonCode/KaggleGuitar/runs/guitar_kaggle_finetune3/weights/best.pt"
val_dir    = "/Users/jackkilshaw/PythonCode/KaggleGuitar/data/kaggle/Guitar-Detection-2/valid/images"

# --- run predictions ---
model   = YOLO(model_path)
results = model.predict(source=val_dir, conf=0.35, verbose=False)

# --- build truth / prediction lists ---
labels_dir = val_dir.replace("images", "labels")
y_true, y_pred, file_names = [], [], []

for r in results:
    img_name   = os.path.basename(r.path)
    label_file = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

    has_label  = os.path.exists(label_file) and os.path.getsize(label_file) > 0
    pred_box   = len(r.boxes) > 0

    y_true.append(1 if has_label else 0)    # 1 = guitar present
    y_pred.append(1 if pred_box else 0)     # 1 = predicted guitar
    file_names.append(img_name)

# --- compute confusion matrix (labels=[0,1] forces consistent order) ---
cm = confusion_matrix(y_true, y_pred, labels=[0,1])

tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

print("\nPer-Image Confusion Matrix (rows = true, cols = predicted)")
print(cm)
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
print(f"Accuracy : {(tp + tn) / len(y_true):.3f}")
print(f"Precision: {tp / (tp + fp + 1e-9):.3f}")
print(f"Recall   : {tp / (tp + fn + 1e-9):.3f}")

# --- visual heatmap ---
disp = ConfusionMatrixDisplay(cm, display_labels=['Background', 'Guitar'])
disp.plot(cmap='Blues', values_format='d')
plt.title("Per-Image Confusion Matrix (with True Negatives)")
plt.tight_layout()
plt.savefig("/Users/jackkilshaw/PythonCode/KaggleGuitar/runs/guitar_kaggle_finetune3/confusion_matrix_per_image_FIXED.png")
plt.show()

