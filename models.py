# train_classifier.py
import json
import numpy as np
from sklearn.svm import SVC
import joblib
import os

with open("all_embeddings.json", "r") as f:
    emb = json.load(f)

X = np.array(list(emb.values()))

X = np.array([np.array(v).reshape(-1) for v in X])

y = [path.split(os.sep)[-2] for path in emb.keys()]

clf = SVC(gamma='scale', probability=True)
clf.fit(X, y)

joblib.dump(clf, "svm_model.pkl")
print("✓ Saved model to svm_model.pkl")
