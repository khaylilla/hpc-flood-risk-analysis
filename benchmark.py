from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
REPEATS = 3


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"File dataset train tidak ditemukan: {train_path}")
    return pd.read_csv(train_path)


def profile_dataset(train_df: pd.DataFrame) -> dict:
    profile = {
        "n_rows": int(train_df.shape[0]),
        "n_columns": int(train_df.shape[1]),
        "duplicate_rows": int(train_df.duplicated().sum()),
        "missing_values": train_df.isna().sum().to_dict(),
        "dtypes": train_df.dtypes.astype(str).to_dict(),
        "target_distribution": train_df["Survived"].value_counts(normalize=True).sort_index().to_dict(),
    }
    return profile


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        try:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ])
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear")
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    raise ValueError(f"Model tidak dikenal: {model_name}")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def benchmark_one_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    preprocessor = make_preprocessor(X_train)
    model = make_model(model_name)

    preprocess_start = time.perf_counter()
    X_train_processed = preprocessor.fit_transform(X_train)
    preprocess_duration = time.perf_counter() - preprocess_start

    train_start = time.perf_counter()
    model.fit(X_train_processed, y_train)
    train_duration = time.perf_counter() - train_start

    valid_start = time.perf_counter()
    X_valid_processed = preprocessor.transform(X_valid)
    y_pred = model.predict(X_valid_processed)
    valid_duration = time.perf_counter() - valid_start

    metrics = compute_metrics(y_valid, y_pred)
    record = {
        "model_name": model_name,
        "preprocessing_time_seconds": preprocess_duration,
        "training_time_seconds": train_duration,
        "inference_time_seconds": valid_duration,
        "total_time_seconds": preprocess_duration + train_duration + valid_duration,
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_valid)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }
    return record


def run_benchmark() -> pd.DataFrame:
    ensure_output_dir()
    train_df = load_dataset()
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    X_train_full, X_valid, y_train_full, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    results = []
    for fraction in DATA_FRACTIONS:
        if np.isclose(fraction, 1.0):
            X_subset = X_train_full.copy()
            y_subset = y_train_full.copy()
        else:
            X_subset, _, y_subset, _ = train_test_split(
                X_train_full,
                y_train_full,
                train_size=fraction,
                stratify=y_train_full,
                random_state=SEED,
            )

        for model_name in ["Logistic Regression", "Random Forest"]:
            for repeat_idx in range(REPEATS):
                result = benchmark_one_model(model_name, X_subset, y_subset, X_valid, y_valid)
                result["data_fraction"] = fraction
                result["repeat"] = repeat_idx + 1
                results.append(result)

    benchmark_df = pd.DataFrame(results)
    benchmark_df = benchmark_df.sort_values(["data_fraction", "model_name", "repeat"]).reset_index(drop=True)
    benchmark_df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
    return benchmark_df


def save_runtime_chart(benchmark_df: pd.DataFrame) -> None:
    summary_df = benchmark_df.groupby(["data_fraction", "model_name"], as_index=False).agg(
        total_time_seconds=("total_time_seconds", "mean"),
        preprocessing_time_seconds=("preprocessing_time_seconds", "mean"),
        training_time_seconds=("training_time_seconds", "mean"),
    )

    plt.figure(figsize=(10, 6))
    for model_name in sorted(summary_df["model_name"].unique()):
        model_subset = summary_df[summary_df["model_name"] == model_name].sort_values("data_fraction")
        plt.plot(model_subset["data_fraction"], model_subset["total_time_seconds"], marker="o", label=model_name)

    plt.title("Perbandingan Waktu Eksekusi Berdasarkan Ukuran Dataset")
    plt.xlabel("Proporsi Data Training")
    plt.ylabel("Waktu Total (detik)")
    plt.xticks(DATA_FRACTIONS)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_comparison.png", dpi=300)
    plt.close()


def save_metrics_chart(benchmark_df: pd.DataFrame) -> None:
    summary_df = benchmark_df.groupby(["model_name"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
    )
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(summary_df))
    width = 0.18

    plt.figure(figsize=(10, 6))
    for idx, metric in enumerate(metric_names):
        values = summary_df[metric].to_numpy()
        plt.bar(x + (idx - 1.5) * width, values, width=width, label=metric.upper())

    plt.xticks(x, summary_df["model_name"].tolist())
    plt.title("Perbandingan Metrik Evaluasi Rata-Rata")
    plt.xlabel("Model")
    plt.ylabel("Skor")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=300)
    plt.close()


def save_summary_text(benchmark_df: pd.DataFrame, profile: dict) -> None:
    summary_df = benchmark_df.groupby(["data_fraction", "model_name"], as_index=False).agg(
        total_time_seconds=("total_time_seconds", "mean"),
        accuracy=("accuracy", "mean"),
        f1=("f1", "mean"),
    )
    best_runtime = summary_df.loc[summary_df["total_time_seconds"].idxmin()]
    best_model = best_runtime["model_name"]
    best_fraction = best_runtime["data_fraction"]

    summary_lines = [
        "Ringkasan Insight Benchmark Titanic",
        "===================================",
        "",
        f"Jumlah baris dataset training: {profile['n_rows']}",
        f"Jumlah kolom: {profile['n_columns']}",
        f"Duplikasi baris: {profile['duplicate_rows']}",
        f"Distribusi target Survived: {profile['target_distribution']}",
        f"Missing values: {profile['missing_values']}",
        "",
        "Hasil benchmark menunjukkan bahwa ukuran data memengaruhi waktu preprocessing dan training secara nyata.",
        f"Model dengan waktu total terendah adalah {best_model} pada proporsi data {best_fraction}.",
        "",
        "Catatan penting:",
        "- Benchmark ini dilakukan pada dataset kecil dan komputer pribadi.",
        "- Hasil ini bukan bukti langsung performa superkomputer atau HPC.",
        "- Ini lebih tepat dipahami sebagai demonstrasi konseptual skalabilitas komputasi dan biaya komputasi ML.",
        "",
        "Metrik paling penting dari hasil eksperimen dipetakan otomatis dalam file CSV dan grafik PNG yang tersimpan di folder output/.",
    ]

    (OUTPUT_DIR / "benchmark_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    train_df = load_dataset()
    profile = profile_dataset(train_df)
    benchmark_df = run_benchmark()
    save_runtime_chart(benchmark_df)
    save_metrics_chart(benchmark_df)
    save_summary_text(benchmark_df, profile)

    print("Profil dataset Titanic:")
    print(profile)
    print("\nHasil benchmark disimpan ke folder output/:")
    for item in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {item.name}")

    print("\nPreview hasil benchmark:")
    print(benchmark_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
