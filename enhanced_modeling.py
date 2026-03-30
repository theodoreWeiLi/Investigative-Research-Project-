import glob
import json
import os
import warnings
import argparse

import librosa
import numpy as np
import pandas as pd
import wfdb
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
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


class ReliefFSelector(BaseEstimator, TransformerMixin):
    """Feature selector wrapper for skrebate.ReliefF so it can be used in sklearn pipelines."""

    def __init__(self, n_features_to_select=30, n_neighbors=50):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.selector_ = None
        self.support_indices_ = None

    def fit(self, X, y):
        from skrebate import ReliefF

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        n_features = X_arr.shape[1]
        n_select = min(self.n_features_to_select, n_features)
        n_neighbors = min(self.n_neighbors, max(1, len(y_arr) - 1))

        self.selector_ = ReliefF(
            n_features_to_select=n_select,
            n_neighbors=n_neighbors,
        )
        self.selector_.fit(X_arr, y_arr)
        self.support_indices_ = np.argsort(self.selector_.feature_importances_)[::-1][:n_select]
        return self

    def transform(self, X):
        if self.support_indices_ is None:
            return X
        return X[:, self.support_indices_]


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


def _extract_standard_feature_block(y: np.ndarray, sr: float, prefix: str = "") -> dict:
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = {
        f"{prefix}centroid_mean": float(np.mean(centroid)),
        f"{prefix}centroid_std": float(np.std(centroid)),
        f"{prefix}bandwidth_mean": float(np.mean(bandwidth)),
        f"{prefix}bandwidth_std": float(np.std(bandwidth)),
        f"{prefix}zcr_mean": float(np.mean(zcr)),
        f"{prefix}zcr_std": float(np.std(zcr)),
    }

    for i in range(13):
        features[f"{prefix}mfcc_{i + 1}_mean"] = float(np.mean(mfccs[i]))
        features[f"{prefix}mfcc_{i + 1}_std"] = float(np.std(mfccs[i]))

    return features


def _decompose_signal(y: np.ndarray, method: str = "ceemdan", max_imfs: int = 3):
    from PyEMD import CEEMDAN, EMD

    if method == "ceemdan":
        decomposer = CEEMDAN(trials=20, random_seed=42)
        imfs = decomposer.ceemdan(y)
    else:
        decomposer = EMD()
        imfs = decomposer.emd(y)

    if imfs is None or len(imfs) == 0:
        return []
    return imfs[:max_imfs]


def extract_features(
    record_id: str,
    data_dir: str,
    use_decomposition: bool = False,
    decomposition_method: str = "ceemdan",
    max_imfs: int = 3,
) -> dict | None:
    record_path = os.path.join(data_dir, record_id)
    try:
        record = wfdb.rdrecord(record_path)
        y = record.p_signal[:, 0]
        sr = record.fs

        features = {"ID": record_id}
        features.update(_extract_standard_feature_block(y, sr, prefix="raw_" if use_decomposition else ""))

        if use_decomposition:
            imfs = _decompose_signal(y, method=decomposition_method, max_imfs=max_imfs)
            for idx, imf in enumerate(imfs, start=1):
                features.update(_extract_standard_feature_block(imf, sr, prefix=f"imf{idx}_"))

        return features
    except Exception:
        return None


def build_dataset(
    data_dir: str,
    use_decomposition: bool = False,
    decomposition_method: str = "ceemdan",
    max_imfs: int = 3,
    max_records: int | None = None,
    feature_cache_path: str | None = None,
) -> pd.DataFrame:
    if feature_cache_path and os.path.exists(feature_cache_path):
        return pd.read_csv(feature_cache_path)

    df_meta = parse_metadata(data_dir)
    if max_records is not None:
        df_meta = df_meta.head(max_records)

    all_features = []
    total = len(df_meta)
    for i, record_id in enumerate(df_meta["ID"], start=1):
        if i % 20 == 0 or i == total:
            print(f"Feature extraction progress: {i}/{total}")

        feat = extract_features(
            record_id,
            data_dir,
            use_decomposition=use_decomposition,
            decomposition_method=decomposition_method,
            max_imfs=max_imfs,
        )
        if feat is not None:
            all_features.append(feat)

    df_features = pd.DataFrame(all_features)
    df = pd.merge(df_meta, df_features, on="ID", how="inner")

    if feature_cache_path:
        os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
        df.to_csv(feature_cache_path, index=False)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced VOICED modeling pipeline")
    parser.add_argument("--use-decomposition", action="store_true", help="Enable EMD/CEEMDAN decomposition features")
    parser.add_argument("--decomposition-method", choices=["emd", "ceemdan"], default="ceemdan")
    parser.add_argument("--max-imfs", type=int, default=3)
    parser.add_argument("--use-relieff", action="store_true", help="Enable ReliefF feature selection")
    parser.add_argument("--max-records", type=int, default=None, help="Optional debug mode to run on subset")
    parser.add_argument("--feature-cache", type=str, default=None, help="Optional CSV cache path for extracted features")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = "voice-icar-federico-ii-database-1.0.0"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Building dataset with audio + metadata features...")
    if args.use_decomposition:
        print(f"Decomposition enabled: method={args.decomposition_method}, max_imfs={args.max_imfs}")

    df = build_dataset(
        data_dir,
        use_decomposition=args.use_decomposition,
        decomposition_method=args.decomposition_method,
        max_imfs=args.max_imfs,
        max_records=args.max_records,
        feature_cache_path=args.feature_cache,
    )
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
        "SVM_Cubic": (
            SVC(
                kernel="poly",
                degree=3,
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            {
                "clf__C": [0.5, 1.0, 5.0, 10.0],
                "clf__gamma": ["scale", 0.01, 0.1],
                "clf__coef0": [0.0, 1.0],
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
    minority_count = int((y_train == 0).sum())
    majority_count = int((y_train == 1).sum())
    minority_count = min(minority_count, majority_count)

    if minority_count >= 25:
        cv_splits = 5
    elif minority_count >= 10:
        cv_splits = 3
    else:
        cv_splits = 2

    if minority_count >= 6:
        sampler = SMOTE(random_state=42, k_neighbors=2)
        sampler_name = "SMOTE(k_neighbors=2)"
    else:
        sampler = RandomOverSampler(random_state=42)
        sampler_name = "RandomOverSampler"

    print(f"Sampling strategy: {sampler_name}, CV folds: {cv_splits}")

    for model_name, (estimator, param_grid) in models.items():
        print(f"\nTraining {model_name}...")

        steps = [("prep", preprocessor)]
        if args.use_relieff:
            steps.append(("selector", ReliefFSelector(n_features_to_select=30, n_neighbors=50)))
            param_grid = dict(param_grid)
            param_grid["selector__n_features_to_select"] = [20, 30, 40]
        steps.append(("smote", sampler))
        steps.append(("clf", estimator))

        pipeline = ImbPipeline(steps=steps)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=cv_splits,
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
    result_suffix = "baseline"
    if args.use_decomposition and args.use_relieff:
        result_suffix = f"{args.decomposition_method}_relieff"
    elif args.use_decomposition:
        result_suffix = args.decomposition_method
    elif args.use_relieff:
        result_suffix = "relieff"

    result_csv = os.path.join(output_dir, f"enhanced_model_results_{result_suffix}.csv")
    result_json = os.path.join(output_dir, f"best_params_{result_suffix}.json")

    df_results.to_csv(result_csv, index=False)

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nSaved results to {result_csv}")
    print(f"Saved hyperparameters to {result_json}")
    print("\nTop models by specificity:")
    print(df_results[["model", "specificity", "sensitivity", "f1", "accuracy"]].head(5))


if __name__ == "__main__":
    main()
