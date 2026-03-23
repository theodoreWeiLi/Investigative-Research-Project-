import glob
import json
import os
import warnings

import librosa
import numpy as np
import pandas as pd
import wfdb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)


def parse_metadata(data_dir: str) -> pd.DataFrame:
    metadata_list = []
    info_files = sorted(glob.glob(os.path.join(data_dir, "*-info.txt")))

    keys_of_interest = {
        "Age": "Age",
        "Gender": "Gender",
        "Diagnosis": "Diagnosis",
        "Smoker": "Smoker",
        "Voice Handicap Index (VHI) Score": "VHI",
        "Reflux Symptom Index (RSI) Score": "RSI",
    }

    for file_path in info_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        patient_data = {
            "ID": os.path.basename(file_path).replace("-info.txt", ""),
            "Age": np.nan,
            "Gender": np.nan,
            "Diagnosis": np.nan,
            "Smoker": np.nan,
            "VHI": np.nan,
            "RSI": np.nan,
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                parts = line.split(":", 1)
            elif "\t" in line:
                parts = line.split("\t", 1)
            else:
                continue

            if len(parts) != 2:
                continue

            raw_key = parts[0].strip()
            raw_val = parts[1].strip()
            if raw_key in keys_of_interest:
                patient_data[keys_of_interest[raw_key]] = raw_val

        metadata_list.append(patient_data)

    df_meta = pd.DataFrame(metadata_list)
    df_meta.replace("NU", np.nan, inplace=True)
    df_meta["Age"] = pd.to_numeric(df_meta["Age"], errors="coerce")
    df_meta["VHI"] = pd.to_numeric(df_meta["VHI"], errors="coerce")
    df_meta["RSI"] = pd.to_numeric(df_meta["RSI"], errors="coerce")

    # Healthy = 0, Pathological = 1
    df_meta["Label"] = df_meta["Diagnosis"].apply(
        lambda x: 0 if str(x).strip().lower() == "healthy" else 1
    )
    return df_meta


def extract_features(record_id: str, data_dir: str) -> dict | None:
    record_path = os.path.join(data_dir, record_id)
    try:
        record = wfdb.rdrecord(record_path)
        y = record.p_signal[:, 0]
        sr = record.fs

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        features = {
            "ID": record_id,
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "bandwidth_std": float(np.std(bandwidth)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
        }

        for i in range(13):
            features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i + 1}_std"] = float(np.std(mfccs[i]))

        return features
    except Exception:
        return None


def build_dataset(data_dir: str) -> pd.DataFrame:
    df_meta = parse_metadata(data_dir)
    all_features = []
    for record_id in df_meta["ID"]:
        feat = extract_features(record_id, data_dir)
        if feat is not None:
            all_features.append(feat)

    df_features = pd.DataFrame(all_features)
    df = pd.merge(df_meta, df_features, on="ID", how="inner")
    return df


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, min_sensitivity: float = 0.70) -> float:
    best_threshold = 0.5
    best_specificity = -1.0

    for threshold in np.linspace(0.1, 0.9, 81):
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sensitivity >= min_sensitivity and specificity > best_specificity:
            best_specificity = specificity
            best_threshold = float(threshold)

    return best_threshold


def evaluate_with_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_score_vector(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return 1 / (1 + np.exp(-raw))
    return model.predict(x)


def main():
    data_dir = "voice-icar-federico-ii-database-1.0.0"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Building dataset with audio + metadata features...")
    df = build_dataset(data_dir)
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Class distribution:")
    print(df["Label"].value_counts())

    feature_cols = [
        c
        for c in df.columns
        if c not in ["ID", "Diagnosis", "Label"]
    ]

    X = df[feature_cols]
    y = df["Label"].astype(int)

    # Two-stage split: train / val / test = 60 / 20 / 20
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    numeric_cols = [c for c in feature_cols if c not in ["Gender", "Smoker"]]
    categorical_cols = [c for c in ["Gender", "Smoker"] if c in feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            {
                "clf__C": [0.1, 1.0, 10.0],
            },
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
            {
                "clf__C": [0.5, 1.0, 5.0],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [4, 6, 8, 12],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "MLP": (
            MLPClassifier(max_iter=1200, early_stopping=True, random_state=42),
            {
                "clf__hidden_layer_sizes": [(64,), (128,), (128, 64)],
                "clf__alpha": [1e-4, 1e-3, 1e-2],
                "clf__learning_rate_init": [1e-3, 5e-4],
            },
        ),
    }

    # Optional advanced models
    try:
        from xgboost import XGBClassifier

        ratio = (y_train == 1).sum() / max((y_train == 0).sum(), 1)
        models["XGBoost"] = (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
            ),
            {
                "clf__n_estimators": [150, 250],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__scale_pos_weight": [ratio],
            },
        )
    except Exception:
        print("XGBoost not available; skipping.")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = (
            CatBoostClassifier(verbose=0, random_seed=42),
            {
                "clf__depth": [4, 6, 8],
                "clf__learning_rate": [0.03, 0.1],
                "clf__iterations": [200, 400],
            },
        )
    except Exception:
        print("CatBoost not available; skipping.")

    results = []
    best_params = {}

    print("\nRunning model selection and threshold tuning...")
    for model_name, (estimator, param_grid) in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = ImbPipeline(
            steps=[
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_params[model_name] = search.best_params_

        val_scores = get_score_vector(best_model, X_val)
        best_threshold = choose_threshold(
            y_true=y_val.to_numpy(),
            scores=val_scores,
            min_sensitivity=0.70,
        )

        test_scores = get_score_vector(best_model, X_test)
        metrics = evaluate_with_threshold(
            y_true=y_test.to_numpy(),
            scores=test_scores,
            threshold=best_threshold,
        )

        metrics["model"] = model_name
        metrics["threshold"] = best_threshold
        results.append(metrics)

        print(
            f"{model_name}: accuracy={metrics['accuracy']:.3f}, "
            f"sensitivity={metrics['sensitivity']:.3f}, "
            f"specificity={metrics['specificity']:.3f}"
        )

    df_results = pd.DataFrame(results).sort_values(
        by=["specificity", "sensitivity", "f1"], ascending=False
    )
    df_results.to_csv(os.path.join(output_dir, "enhanced_model_results.csv"), index=False)

    with open(os.path.join(output_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nSaved results to outputs/enhanced_model_results.csv")
    print("Saved hyperparameters to outputs/best_params.json")
    print("\nTop models by specificity:")
    print(df_results[["model", "specificity", "sensitivity", "f1", "accuracy"]].head(5))


if __name__ == "__main__":
    main()
