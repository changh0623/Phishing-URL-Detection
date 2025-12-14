import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import random

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 1) Dataset loader (HF)
# -----------------------------
def load_urls(dataset_name="shawhin/phishing-site-classification", max_n=20000):
    """
    Dataset: shawhin/phishing-site-classification (Hugging Face)
      - url/text: URL string
      - label: int (0 benign, 1 phishing)

    This loader:
      1) normalizes column names
      2) finds the URL and label columns
      3) drops NA + trims strings + drops duplicate URLs
      4) balances classes by downsampling to equal counts (0/1)
      5) optionally caps size via max_n
    """
    ds = load_dataset(dataset_name)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    # 1) Normalize column names: lowercase + strip spaces
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) Find the URL column (it can be "url" or "text" depending on the dataset)
    if "url" in df.columns:
        url_col = "url"
    elif "text" in df.columns:
        url_col = "text"
    else:
        # Safety fallback when switching datasets
        candidates = [c for c in df.columns if "url" in c]
        if not candidates:
            raise ValueError(f"URL column not found. columns={list(df.columns)}")
        url_col = candidates[0]

    # 3) Find the label column
    if "label" in df.columns:
        label_col = "label"
    else:
        candidates = [c for c in df.columns if "label" in c or "target" in c or "class" in c]
        if not candidates:
            raise ValueError(f"Label column not found. columns={list(df.columns)}")
        label_col = candidates[0]

    df = df[[url_col, label_col]].dropna()
    df = df.rename(columns={url_col: "url", label_col: "label"})

    df["url"] = df["url"].astype(str)
    df["url"] = df["url"].str.strip()
    df["label"] = df["label"].astype(int)

    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # --- Balance classes (make 0/1 counts equal) ---
    min_count = df["label"].value_counts().min()
    target_per_class = min(min_count, max_n // 2)

    df0 = df[df["label"] == 0].sample(n=target_per_class, random_state=SEED)
    df1 = df[df["label"] == 1].sample(n=target_per_class, random_state=SEED)
    df = pd.concat([df0, df1]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return df
       
# -----------------------------
# 2) Naïve baseline (heuristic)
# -----------------------------
SUSPICIOUS_TOKENS = ["login", "secure", "verify", "update", "account", "payment", "bank"]
def heuristic_score(url: str) -> int:
    u = url.lower()
    score = 0

    # length
    if len(u) > 50:
        score += 1

    # special characters
    if "@" in u:
        score += 2
    if u.count("-") >= 2:
        score += 1
    if u.count(".") >= 4:
        score += 1

    # digits (often in obfuscated hosts/paths)
    digit_ratio = sum(ch.isdigit() for ch in u) / max(1, len(u))
    if digit_ratio > 0.12:
        score += 1

    # punycode (internationalized domain trick)
    if "xn--" in u:
        score += 2

    # token hints
    if any(t in u for t in SUSPICIOUS_TOKENS):
        score += 1

    return score

def predict_from_scores(scores, threshold: int):
    return [1 if s >= threshold else 0 for s in scores]

# -----------------------------
# 3) URL feature baseline (optional stronger-naïve)
# -----------------------------
def featurize_urls(urls):
    feats = []
    for url in urls:
        u = url.lower()
        feats.append([
            len(u),
            u.count("."),
            u.count("-"),
            u.count("@"),
            u.count("?"),
            u.count("="),
            u.count("%"),
            int(u.startswith("https://")),
            int("xn--" in u),
            sum(ch.isdigit() for ch in u),
        ])
    cols = ["len","dots","hyphens","ats","qmarks","equals","percents","is_https","has_punycode","num_digits"]
    return pd.DataFrame(feats, columns=cols)

def main():
    os.makedirs("outputs", exist_ok=True)
    print("Loading dataset...")
    df = load_urls(dataset_name="shawhin/phishing-site-classification", max_n=3000)
    print("Dataset size:", len(df))
    print(df.head(3))
    print("Label counts:\n", df["label"].value_counts())
    print("Label ratio:\n", df["label"].value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        df["url"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    
    # ---- Baseline 1: heuristic (tune threshold on train -> evaluate on test)
    print("\n=== Baseline (heuristic, threshold tuned on train) ===")

    # split train into (train_sub, val) for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    # compute heuristic scores (train/val only for tuning)
    tr_scores = [heuristic_score(u) for u in X_tr]
    val_scores = [heuristic_score(u) for u in X_val]

    # score range is small and bounded; use only train/val to define candidates
    max_score = max(tr_scores + val_scores)
    thresholds = list(range(1, max_score + 1))

    best_thr = None
    best_f1 = -1.0
    best_recall = -1.0
    tuning_rows = []

    for thr in thresholds:
        y_val_pred = predict_from_scores(val_scores, thr)

        if len(set(y_val_pred)) < 2:
            continue

        f1_1 = f1_score(y_val, y_val_pred, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_val, y_val_pred, pos_label=1, zero_division=0)

        tuning_rows.append({"threshold": thr, "f1_1": f1_1, "recall_1": rec_1})

        # primary: maximize F1(1), tiebreak: higher recall(1)
        if (f1_1 > best_f1) or (f1_1 == best_f1 and rec_1 > best_recall):
            best_f1 = f1_1
            best_recall = rec_1
            best_thr = thr
    
    if best_thr is None:
        best_thr = 1
    
    # evaluate test AFTER choosing best_thr
    test_scores = [heuristic_score(u) for u in X_test]


    print(f"[Baseline1] Best threshold on val: {best_thr} (F1_1={best_f1:.4f}, Recall_1={best_recall:.4f})")

    # evaluate on test using best_thr
    y_pred_b1 = predict_from_scores(test_scores, best_thr)

    print(classification_report(y_test, y_pred_b1, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_b1))

    report_b1 = classification_report(y_test, y_pred_b1, output_dict=True, zero_division=0)
    cm_b1 = confusion_matrix(y_test, y_pred_b1)

    # save tuning table + chosen threshold
    pd.DataFrame(tuning_rows).to_csv("outputs/baseline1_threshold_tuning.csv", index=False)
    with open("outputs/baseline1_best_threshold.txt", "w") as f:
        f.write(str(best_thr))

    # ---- Baseline 2: engineered features + LogisticRegression (still lightweight/naïve)
    print("\n=== Baseline (handcrafted features + LR) ===")
    Xtr_f = featurize_urls(X_train.tolist())
    Xte_f = featurize_urls(X_test.tolist())
    lr_feat = LogisticRegression(max_iter=2000, random_state=SEED)
    lr_feat.fit(Xtr_f, y_train)
    y_pred_b2 = lr_feat.predict(Xte_f)
    print(classification_report(y_test, y_pred_b2, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_b2))
    report_b2 = classification_report(y_test, y_pred_b2, output_dict=True, zero_division=0)
    cm_b2 = confusion_matrix(y_test, y_pred_b2)

    # ---- AI Pipeline: sentence-transformers embedding + LR
    print("\n=== AI Pipeline (SentenceTransformer embeddings + LR) ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name, device=device)
    Xtr_emb = embedder.encode(X_train.tolist(), show_progress_bar=True)
    Xte_emb = embedder.encode(X_test.tolist(), show_progress_bar=True)

    clf = LogisticRegression(max_iter=3000, random_state=SEED)
    clf.fit(Xtr_emb, y_train)
    y_pred_ai = clf.predict(Xte_emb)

    print(classification_report(y_test, y_pred_ai, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_ai))
    report_ai = classification_report(y_test, y_pred_ai, output_dict=True, zero_division=0)
    cm_ai = confusion_matrix(y_test, y_pred_ai)

    # ---- Qualitative examples (at least 3)
    print("\n=== Qualitative examples (Baseline1 vs AI) ===")
    test_df = pd.DataFrame({"url": X_test.values, "y": y_test.values})
    test_df["b1"] = y_pred_b1
    test_df["ai"] = y_pred_ai

    ex_good = test_df[(test_df["b1"] != test_df["y"]) & (test_df["ai"] == test_df["y"])].head(3)
    ex_bad = test_df[(test_df["b1"] == test_df["y"]) & (test_df["ai"] != test_df["y"])].head(1)
    ex_hard = test_df[(test_df["b1"] != test_df["y"]) & (test_df["ai"] != test_df["y"])].head(1)

    examples = pd.concat([ex_good, ex_bad, ex_hard]).drop_duplicates()

    if len(examples) == 0:
        print("No differing examples found (models identical on test set).")
    else:
        print(examples.to_string(index=False))

    # ---- Save outputs (csv + json)
    # 1) examples
    examples.to_csv("outputs/qualitative_examples.csv", index=False)

    # 2) confusion matrices
    pd.DataFrame(cm_b1, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv("outputs/cm_baseline1.csv")
    pd.DataFrame(cm_b2, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv("outputs/cm_baseline2.csv")
    pd.DataFrame(cm_ai, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv("outputs/cm_ai.csv")

    # 3) summary metrics table
    def pick_metrics(r):
        # r['accuracy'] is float, r['1'] is dict if labels are "0"/"1" strings
        return {
            "accuracy": r.get("accuracy", None),
            "precision_1": r.get("1", {}).get("precision", None),
            "recall_1": r.get("1", {}).get("recall", None),
            "f1_1": r.get("1", {}).get("f1-score", None),
        }

    rows = [
        {"method": "baseline_heuristic", **pick_metrics(report_b1)},
        {"method": "baseline_features_lr", **pick_metrics(report_b2)},
        {"method": "ai_minilm_embed_lr", **pick_metrics(report_ai)},
    ]
    pd.DataFrame(rows).to_csv("outputs/results_summary.csv", index=False)

    # 4) also save raw reports
    with open("outputs/reports.json", "w") as f:
        json.dump({"baseline1": report_b1, "baseline2": report_b2, "ai": report_ai}, f, indent=2)

    # 5) simple bar plot (accuracy)
    accs = [report_b1["accuracy"], report_b2["accuracy"], report_ai["accuracy"]]
    labels = ["B1_heuristic", "B2_feat+LR", "AI_embed+LR"]
    plt.figure()
    plt.bar(labels, accs)
    plt.ylim(0, 1.0)
    plt.title("Accuracy by method")
    plt.ylabel("accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("outputs/accuracy_bar.png", dpi=200)
    plt.close()

    print("\n[Saved] outputs/results_summary.csv, outputs/reports.json, outputs/cm_*.csv, outputs/qualitative_examples.csv, outputs/accuracy_bar.png")

if __name__ == "__main__":
    main()
