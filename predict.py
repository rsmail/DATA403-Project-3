# dev_eval.py
import os
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# 1. Load model + DINOv2 + transforms
# --------------------------------------------------

model_svm = joblib.load("svm_model.pkl")

dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2.to(device)

transform = T.Compose([
    T.Resize(244),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def embed(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = dinov2(x)[0].cpu().numpy().reshape(1, -1)
    return emb

# --------------------------------------------------
# 2. Evaluate the Dev folder
# --------------------------------------------------

DEV_DIR = os.path.join(os.getcwd(), "Dev")
classes = ["Alex", "Kelly"]

y_true = []
y_pred = []

for label in classes:
    folder_path = os.path.join(DEV_DIR, label)
    print(f"Evaluating folder: {folder_path}")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)
            emb = embed(path)
            pred = model_svm.predict(emb)[0]

            y_true.append(label)
            y_pred.append(pred)

# --------------------------------------------------
# 3. Compute accuracy + confusion matrix
# --------------------------------------------------

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=classes)

print("\n===========================")
print("DEV SET EVALUATION RESULTS")
print("===========================\n")
print("Overall Accuracy:", acc)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(classes)
print(cm)
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))
