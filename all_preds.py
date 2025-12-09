import json
import numpy as np
import pandas as pd
import joblib
import os

with open("all_embeddings.json", "r") as f:
    emb = json.load(f)

paths = list(emb.keys())
X = np.array([np.array(v).reshape(-1) for v in emb.values()])
clf = joblib.load("svm_model.pkl")
probs = clf.predict_proba(X) 

# pred probabilities of Alex
classes = clf.classes_  
alex_idx = np.where(classes == "Alex")[0][0]

alex_probs = probs[:, alex_idx]  # P(image is Alex)
filenames = []
for p in paths:
    base = os.path.basename(p)              
    name_only = os.path.splitext(base)[0]   
    filenames.append(name_only)
preds_df = pd.DataFrame({
    "filename": filenames,
    "model_pred_pct": alex_probs
})
preds_df.to_csv("model_predictions.csv", index=False)
