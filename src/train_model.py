import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import joblib
import os

DATA_DIR = "../data/"
LABELS_FILE = os.path.join(DATA_DIR, "PV_Diagnostic_Final_Labels.csv")
MODEL_SAVE_PATH = "../models/pv_degradation_model.pkl"


def load_data():
    print("Loading labels file (separator = ';')...")
    df = pd.read_csv(LABELS_FILE, sep=";")
    print(f"Shape: {df.shape}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df["Label_S1"] = df["Label_S1"].astype(str).str.strip()
    df["Label_S2"] = df["Label_S2"].astype(str).str.strip()

    print("\nLabel_S1 counts:", df["Label_S1"].value_counts().to_dict())
    print("Label_S2 counts:", df["Label_S2"].value_counts().to_dict())

    return df


def train_degradation_model(df, string_num):
    print(f"\n{'='*60}")
    print(f"MODEL: String {string_num} | Normal vs Degradation ONLY")
    print(f"{'='*60}")

    label_col = f"Label_S{string_num}"
    V = f"V{string_num}"
    I = f"I{string_num}"
    T = f"T{string_num}"

    # Keep only Normal and Degradation rows
    keep = df[df[label_col].isin(["Normal", "Degradation"])].copy()
    print(f"Rows after filtering (Normal + Degradation only): {len(keep)}")
    print(f"  Normal:      {(keep[label_col] == 'Normal').sum()}")
    print(f"  Degradation: {(keep[label_col] == 'Degradation').sum()}")

    # Features
    features = keep[[V, I, "Irr", T]].copy()
    features["Power"] = features[V] * features[I]
    features["PerformanceRatio"] = features["Power"] / (features["Irr"] + 1e-6)
    features["Resistance"] = features[V] / (features[I] + 1e-6)

    # Binary label: 0 = Normal, 1 = Degradation
    y = (keep[label_col] == "Degradation").astype(int)

    # Stratified split — preserves class ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train Normal/Degraded: {(y_train == 0).sum()} / {(y_train == 1).sum()}")
    print(f"Test  Normal/Degraded: {(y_test == 0).sum()} / {(y_test == 1).sum()}")

    # Baseline: always predict the majority class
    majority = int(y_train.mode()[0])
    baseline_acc = (y_test == majority).mean() * 100
    print(f"\nBaseline accuracy (always predict majority): {baseline_acc:.2f}%")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%  "
          f"(vs baseline {baseline_acc:.2f}%)")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nConfusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["Actual Normal", "Actual Degraded"],
        columns=["Predicted Normal", "Predicted Degraded"],
    ))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Degraded"],
                                zero_division=0))

    importances = pd.Series(
        model.feature_importances_, index=features.columns
    ).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.to_string())

    return model


if __name__ == "__main__":
    df = load_data()
    model_s1 = train_degradation_model(df, 1)
    model_s2 = train_degradation_model(df, 2)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump({"s1": model_s1, "s2": model_s2}, MODEL_SAVE_PATH)
    print(f"\n[OK] Models saved to {MODEL_SAVE_PATH}")