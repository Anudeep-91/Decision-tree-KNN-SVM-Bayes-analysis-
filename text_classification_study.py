#!/usr/bin/env python3
"""
=============================================================================
  Text Classification Comparative Study — 20 Newsgroups Benchmark
=============================================================================
This script implements an end-to-end comparative study across four classic
machine learning classifiers:
  1. Decision Tree (Gini index)
  2. K-Nearest Neighbors (KNN, k=5)
  3. Support Vector Machine (Linear SVM)
  4. Naïve Bayes (Multinomial NB)

Pipeline:
  - Loads 5 distinct categories from the 20 Newsgroups dataset.
  - Performs stratified 80/20 train/test split.
  - Extracts TF-IDF representations (fit on train only).
  - Trains and evaluates all four models on identical splits.
  - Generates classification reports, metric tables, and confusion matrices.
  - Conducts a "What-If" sensitivity experiment by sweeping TF-IDF max_features.
  - Automatically exports all visual artifacts and a console summary report.
=============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Set Matplotlib backend before importing pyplot to guarantee headless execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES = [
    'sci.med',
    'sci.space',
    'rec.sport.baseball',
    'talk.politics.mideast',
    'comp.graphics'
]

DEFAULT_WHAT_IF_FEATURES = [500, 1000, 2500, 5000, 10000, 20000]
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_MAX_FEATURES = 10000


class DualLogger:
    """
    Tee-style logger that writes output simultaneously to sys.stdout and a log file.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


def print_banner(title, char="="):
    """Prints a styled section header banner."""
    bar = char * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# Phase 1: Setup & Data Loading
# ---------------------------------------------------------------------------
def load_dataset(categories=DEFAULT_CATEGORIES):
    """
    Loads 20 Newsgroups subset for chosen categories and prints summary stats.
    """
    print_banner("[Phase 1] Loading 20 Newsgroups Dataset")
    print(f"Target categories ({len(categories)}):")
    for cat in categories:
        print(f"  - {cat}")

    print("\nFetching data from scikit-learn...")
    dataset = fetch_20newsgroups(
        subset='all',
        categories=categories,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=()  # Retain standard body and metadata for realistic document classification
    )

    total_docs = len(dataset.data)
    print(f"\nSuccessfully loaded {total_docs:,} total documents.")

    # Class distribution
    print("\nDocument count per class:")
    unique_ids, counts = np.unique(dataset.target, return_counts=True)
    for class_id, count in zip(unique_ids, counts):
        class_name = dataset.target_names[class_id]
        pct = (count / total_docs) * 100
        print(f"  * {class_name:<24}: {count:>4} docs ({pct:5.1f}%)")

    # Sample preview
    print("\nSample Document Preview (Document #0):")
    sample_cat = dataset.target_names[dataset.target[0]]
    sample_text = dataset.data[0].strip()
    preview = "\n".join(sample_text.splitlines()[:10])
    print(f"Category: {sample_cat}")
    print("------------------------------------------------------------")
    print(preview)
    print("... [truncated] ...")
    print("------------------------------------------------------------")

    return dataset


# ---------------------------------------------------------------------------
# Phase 2: Text Preprocessing & Feature Extraction
# ---------------------------------------------------------------------------
def preprocess_and_vectorize(dataset, max_features=DEFAULT_MAX_FEATURES):
    """
    Splits data into train/test sets (80/20 stratified) and fits TF-IDF
    exclusively on the training split to prevent data leakage.
    """
    print_banner("[Phase 2] Preprocessing & TF-IDF Vectorization")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.target
    )

    print(f"Split configuration: {int((1-TEST_SIZE)*100)}% Train / {int(TEST_SIZE*100)}% Test (Stratified)")
    print(f"  - Training documents: {len(X_train_raw):,}")
    print(f"  - Testing documents:  {len(X_test_raw):,}")

    print(f"\nVectorizing with TF-IDF (max_features={max_features:,}, stop_words='english', sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        sublinear_tf=True
    )

    # CRITICAL: Fit solely on training corpus
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    print(f"Feature matrix confirmed dimensions:")
    print(f"  - X_train shape : {X_train_vec.shape}")
    print(f"  - X_test shape  : {X_test_vec.shape}")
    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_):,} tokens")

    return (
        X_train_raw, X_test_raw,
        X_train_vec, X_test_vec,
        y_train, y_test,
        vectorizer
    )


# ---------------------------------------------------------------------------
# Phase 3 & 4: Model Training, Prediction & Evaluation
# ---------------------------------------------------------------------------
def get_model_instances():
    """Returns dictionary of the four classifiers with standard configurations."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            criterion='gini',
            random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            metric='minkowski',
            weights='uniform'
        ),
        "Support Vector Machine": SVC(
            kernel='linear',
            random_state=RANDOM_STATE
        ),
        "Naïve Bayes": MultinomialNB(
            alpha=1.0
        )
    }


def train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test, target_names):
    """
    Trains each classifier, measures latency, calculates metrics, and
    builds full classification reports and confusion matrices.
    """
    print_banner("[Phase 3 & 4] Model Training, Prediction & Evaluation")

    models = get_model_instances()
    metrics_records = []
    confusion_matrices = {}
    classification_reports = {}

    for name, model in models.items():
        print(f"--> Training and evaluating: {name}...")

        # Measure training duration
        t_start_train = time.time()
        model.fit(X_train_vec, y_train)
        t_train = time.time() - t_start_train

        # Measure prediction latency
        t_start_pred = time.time()
        y_pred = model.predict(X_test_vec)
        t_pred = time.time() - t_start_pred

        # Compute core metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Store confusion matrix and report
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

        report = classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0
        )
        classification_reports[name] = report

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Macro)": prec_macro,
            "Recall (Macro)": rec_macro,
            "F1-Score (Macro)": f1_macro,
            "Train Time (s)": round(t_train, 3),
            "Predict Time (s)": round(t_pred, 3)
        })

    # Print individual detailed classification reports
    print("\nDetailed Per-Class Classification Reports:")
    for name, report in classification_reports.items():
        print(f"\n--- {name} ---")
        print(report)

    df_results = pd.DataFrame(metrics_records)
    return df_results, confusion_matrices


# ---------------------------------------------------------------------------
# Phase 5: Comparative Tables and Visualization Exports
# ---------------------------------------------------------------------------
def print_comparative_table(df_results):
    """Prints a styled comparative summary table to console."""
    print_banner("[Phase 5] Comparative Results Table")

    display_df = df_results.copy()
    for col in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print(display_df.to_string(index=False))


def plot_comparative_barchart(df_results, output_path):
    """
    Renders and saves a grouped bar chart comparing all 4 models across
    Accuracy, Precision, Recall, and F1-Score.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)

    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    models = df_results["Model"].tolist()

    x = np.arange(len(models))
    total_metrics = len(metrics)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    palette = ["#2b5c8f", "#d95f02", "#2ca02c", "#7570b3"]

    for i, (metric, color) in enumerate(zip(metrics, palette)):
        offsets = x + (i - (total_metrics - 1) / 2) * bar_width
        values = df_results[metric].values
        rects = ax.bar(offsets, values, width=bar_width, label=metric.replace(" (Macro)", ""), color=color, alpha=0.9)

        # Annotate bar values
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='semibold'
            )

    ax.set_ylabel("Score (0.00 - 1.00)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Head-to-Head Classifier Performance on 20 Newsgroups (5 Categories)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, facecolor='white', shadow=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparative bar chart -> {output_path}")


def plot_confusion_matrices(confusion_matrices, target_names, df_results, output_path):
    """
    Plots a 2x2 grid of confusion matrices annotated with class labels and scores.
    """
    sns.set_theme(style="white", font_scale=0.9)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    short_labels = [name.split('.')[-1] if '.' in name else name for name in target_names]

    cm_cmaps = {
        "Decision Tree": "Blues",
        "K-Nearest Neighbors": "Purples",
        "Support Vector Machine": "Greens",
        "Naïve Bayes": "Oranges"
    }

    f1_map = dict(zip(df_results["Model"], df_results["F1-Score (Macro)"]))

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        cmap = cm_cmaps.get(name, "Blues")
        f1_val = f1_map.get(name, 0.0)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar=False,
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            linewidths=0.5,
            linecolor='lightgray'
        )

        ax.set_title(f"{name} (Macro F1 = {f1_val:.4f})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10, fontweight='bold')
        ax.set_ylabel("True Class", fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    fig.suptitle("Confusion Matrix Comparison Across Classifiers", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrices -> {output_path}")


# ---------------------------------------------------------------------------
# Phase 6: Best Model Identification & Diagnostic Insights
# ---------------------------------------------------------------------------
def print_best_model_verdict(df_results, confusion_matrices, target_names):
    """
    Evaluates top performer and prints diagnostic findings based on misclassifications.
    """
    print_banner("[Phase 6] Best Model Verdict & Diagnostic Insights")

    best_row = df_results.sort_values(by="F1-Score (Macro)", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_f1 = best_row["F1-Score (Macro)"]
    best_acc = best_row["Accuracy"]

    print(f"VERDICT: Top Performing Model is >>> {best_name} <<<")
    print(f"  - Macro F1-Score : {best_f1:.4f}")
    print(f"  - Overall Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  - Training Time   : {best_row['Train Time (s)']:.3f} s")
    print(f"  - Prediction Time : {best_row['Predict Time (s)']:.3f} s")

    print("\nArchitectural & Algorithmic Observations:")
    print("  1. Support Vector Machine (Linear SVM):")
    print("     - Typically achieves superior precision and recall in high-dimensional text spaces.")
    print("     - The linear hyperplane effectively separates sparse TF-IDF vectors where data points")
    print("       are naturally linearly separable due to high dimensionality.")
    print("  2. Multinomial Naïve Bayes:")
    print("     - Delivers near-peak performance with near-zero training latency.")
    print("     - Extremely resilient to irrelevant vocabulary features due to Laplace smoothing.")
    print("  3. Decision Tree:")
    print("     - Suffers from greedy orthogonal splits on sparse feature representations.")
    print("     - Highly prone to variance and overfitting without depth constraints or ensemble averaging.")
    print("  4. K-Nearest Neighbors (KNN):")
    print("     - Severely affected by the 'Curse of Dimensionality' in 10,000-dimensional TF-IDF space,")
    print("       where Euclidean/Minkowski distances between sparse vectors concentrate into uniform distances.")

    print("\nCategory Confusion Diagnostics:")
    for name, cm in confusion_matrices.items():
        total_errors = cm.sum() - np.trace(cm)
        print(f"  - {name:<24}: Total test errors = {total_errors:>3} / {cm.sum():>3} docs")


# ---------------------------------------------------------------------------
# Phase 7: What-If Experiment (TF-IDF max_features Sensitivity)
# ---------------------------------------------------------------------------
def run_what_if_experiment(X_train_raw, y_train, X_test_raw, y_test,
                           feature_counts=DEFAULT_WHAT_IF_FEATURES,
                           output_path=None):
    """
    Conducts a parametric experiment varying TF-IDF vocabulary size (max_features)
    and evaluating the impact on all four classifiers simultaneously.
    """
    print_banner("[Phase 7] What-If Experiment: Sensitivity to TF-IDF max_features")
    print(f"Testing vocabulary size levels: {feature_counts}")

    results_by_feature = []

    for k in feature_counts:
        print(f"\n[Experiment] Evaluating max_features = {k:,}...")

        # Re-vectorize on train only
        vec = TfidfVectorizer(max_features=k, stop_words='english', sublinear_tf=True)
        X_train_k = vec.fit_transform(X_train_raw)
        X_test_k = vec.transform(X_test_raw)

        models = get_model_instances()
        step_entry = {"max_features": k}

        for model_name, model in models.items():
            model.fit(X_train_k, y_train)
            preds = model.predict(X_test_k)
            macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            step_entry[model_name] = macro_f1
            print(f"   {model_name:<24} -> Macro F1 = {macro_f1:.4f}")

        results_by_feature.append(step_entry)

    df_what_if = pd.DataFrame(results_by_feature)

    print("\nWhat-If Experiment Summary Table (Macro F1 vs. max_features):")
    formatted_what_if = df_what_if.copy()
    for col in formatted_what_if.columns:
        if col != "max_features":
            formatted_what_if[col] = formatted_what_if[col].apply(lambda x: f"{x:.4f}")
    print(formatted_what_if.to_string(index=False))

    # Plot Sensitivity Curves
    if output_path:
        sns.set_theme(style="whitegrid", font_scale=1.05)
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

        model_styles = {
            "Decision Tree": {"color": "#e41a1c", "marker": "s", "linestyle": "--"},
            "K-Nearest Neighbors": {"color": "#984ea3", "marker": "^", "linestyle": ":"},
            "Support Vector Machine": {"color": "#377eb8", "marker": "o", "linestyle": "-"},
            "Naïve Bayes": {"color": "#ff7f00", "marker": "D", "linestyle": "-."}
        }

        x_vals = df_what_if["max_features"].tolist()

        for model_name, style in model_styles.items():
            if model_name in df_what_if.columns:
                y_vals = df_what_if[model_name].tolist()
                ax.plot(
                    x_vals,
                    y_vals,
                    label=model_name,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=8,
                    linewidth=2.2,
                    linestyle=style["linestyle"]
                )

        ax.set_title("What-If Experiment: Macro F1-Score vs. TF-IDF Vocabulary Capacity (max_features)",
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("TF-IDF max_features (Vocabulary Size)", fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel("Macro F1-Score", fontsize=11, fontweight='bold', labelpad=10)
        ax.set_xscale('log')
        ax.set_xticks(x_vals)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylim(0.40, 1.00)
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.legend(title="Classifiers", frameon=True, framealpha=0.95, facecolor='white', shadow=True)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"\nSaved what-if sensitivity chart -> {output_path}")

    return df_what_if


# ---------------------------------------------------------------------------
# Main Orchestration Routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Text Classification Comparative Study across Decision Tree, KNN, SVM, and Naive Bayes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated plots and summary report (default: outputs/)"
    )
    parser.add_argument(
        "--skip-what-if",
        action="store_true",
        help="Skip the Phase 7 what-if parameter sensitivity experiment."
    )
    parser.add_argument(
        "--what-if-only",
        action="store_true",
        help="Execute only the Phase 7 what-if experiment without full baseline visual generation."
    )
    parser.add_argument(
        "--what-if-features",
        nargs="+",
        type=int,
        default=DEFAULT_WHAT_IF_FEATURES,
        help="Custom list of max_features integers for the what-if experiment (e.g. --what-if-features 500 2000 10000)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    summary_txt_path = os.path.join(args.output_dir, "results_summary.txt")

    # Initialize dual logger to write both to terminal and results_summary.txt
    logger = DualLogger(summary_txt_path)
    sys.stdout = logger

    start_total_time = time.time()
    print_banner("TEXT CLASSIFICATION COMPARATIVE STUDY — 20 NEWSGROUPS", char="#")
    print(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory    : {os.path.abspath(args.output_dir)}")

    try:
        # Phase 1: Load data
        dataset = load_dataset(DEFAULT_CATEGORIES)
        target_names = dataset.target_names

        # Phase 2: Split and vectorize
        (
            X_train_raw, X_test_raw,
            X_train_vec, X_test_vec,
            y_train, y_test,
            vectorizer
        ) = preprocess_and_vectorize(dataset, max_features=DEFAULT_MAX_FEATURES)

        # Baseline execution (Phases 3-6) unless --what-if-only is specified
        if not args.what_if_only:
            # Phase 3 & 4: Train & evaluate
            df_results, confusion_matrices = train_and_evaluate_models(
                X_train_vec, y_train, X_test_vec, y_test, target_names
            )

            # Phase 5: Print table and generate plots
            print_comparative_table(df_results)

            bar_chart_path = os.path.join(args.output_dir, "comparative_table.png")
            plot_comparative_barchart(df_results, bar_chart_path)

            cm_chart_path = os.path.join(args.output_dir, "confusion_matrices.png")
            plot_confusion_matrices(confusion_matrices, target_names, df_results, cm_chart_path)

            # Phase 6: Best model verdict & insights
            print_best_model_verdict(df_results, confusion_matrices, target_names)

        # Phase 7: What-If Experiment
        if not args.skip_what_if:
            what_if_chart_path = os.path.join(args.output_dir, "what_if_experiment.png")
            run_what_if_experiment(
                X_train_raw, y_train, X_test_raw, y_test,
                feature_counts=args.what_if_features,
                output_path=what_if_chart_path
            )

        total_elapsed = time.time() - start_total_time
        print_banner(f"STUDY EXECUTION COMPLETED in {total_elapsed:.2f} seconds", char="#")
        print("Generated Artifacts:")
        print(f"  [1] Results Summary Log   : {os.path.join(args.output_dir, 'results_summary.txt')}")
        if not args.what_if_only:
            print(f"  [2] Comparative Bar Chart : {os.path.join(args.output_dir, 'comparative_table.png')}")
            print(f"  [3] Confusion Matrices    : {os.path.join(args.output_dir, 'confusion_matrices.png')}")
        if not args.skip_what_if:
            print(f"  [4] What-If Experiment    : {os.path.join(args.output_dir, 'what_if_experiment.png')}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()
