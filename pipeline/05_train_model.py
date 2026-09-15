"""
STEP 5: Train Model

Input:  data/processed/training_data.csv (from step 4)
Output: data/outputs/landslide_rf_model.pkl
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

import config

FEATURE_COLS = config.FEATURE_COLS


def main():
    print("=== STEP 5: Train Model ===")

    if not os.path.exists(config.TRAINING_DATA_CSV):
        raise FileNotFoundError(f"{config.TRAINING_DATA_CSV} not found. Run 04_build_training_data.py first.")

    df = pd.read_csv(config.TRAINING_DATA_CSV)
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Landslide", "Landslide"]))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\n5-Fold Stratified CV ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f}")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n=== Feature Importance ===")
    print(importance)

    joblib.dump(model, config.MODEL_PATH)
    print(f"\nModel saved: {config.MODEL_PATH}")
    print("=== STEP 5 COMPLETE ===\n")


if __name__ == "__main__":
    main()
