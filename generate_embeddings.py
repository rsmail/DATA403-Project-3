# to run this stuff make sure to 
#pip install torch numpy opencv-python scikit-learn
# but embeddings shouldnt need to change for a while
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm.notebook import tqdm
from roboflow import Roboflow
import os


cwd = os.getcwd()

ROOT_DIR = os.path.join(cwd, "train")


labels = {}

for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, file)
            labels[full_path] = folder   

files = labels.keys()
print(labels)


dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)
transform_image = T.Compose([
    T.Resize(244),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

def load_image(img: str) -> torch.Tensor:
    img = Image.open(img).convert("RGB")  
    transformed_img = transform_image(img).unsqueeze(0)
    return transformed_img

def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}
    
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        embeddings = dinov2_vits14(load_image(file).to(device))

        all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings

embeddings = compute_embeddings(files)