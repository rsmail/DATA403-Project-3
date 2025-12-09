# this just cleans the model_predictions csv so that the files match back to our csv
import pandas as pd

df = pd.read_csv("model_predictions.csv")

rows = []

for full_path, prob in zip(df["filename"], df["model_pred_pct"]):
    # normalize separators
    path = str(full_path).replace("\\", "/")
    parts = path.split("/")

    base = parts[-1]          # e.g. "Alex-Image01" or "Kelly-Image27"

    # Try to get label from folder if it exists, otherwise from filename prefix
    label = None
    if len(parts) >= 2:
        candidate = parts[-2]  # maybe "Alex" or "Kelly"
        if candidate.lower() in ["alex", "kelly"]:
            label = candidate

    if label is None:
        if base.lower().startswith("alex"):
            label = "Alex"
        elif base.lower().startswith("kelly"):
            label = "Kelly"

    # Extract digits from file name 
    digits = "".join(ch for ch in base if ch.isdigit())
    img_number = int(digits)
    rows.append([label, img_number, prob])
clean = pd.DataFrame(rows, columns=["label", "img_number", "model_pred_pct"])
clean = clean.sort_values(["label", "img_number"]).reset_index(drop=True)
clean["filename"] = clean.groupby("label").cumcount() + 1
clean.to_csv("clean_model_predictions.csv", index=False)
print(clean.head())