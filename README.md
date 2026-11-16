# Text Classification Comparative Study

An end-to-end comparative study benchmark comparing four foundational machine learning algorithms on real-world text classification using scikit-learn's **20 Newsgroups** dataset.

---

## 📌 Project Overview

This project implements a unified, single-script pipeline that:
1. Loads **~4,700 documents** across 5 distinct topics from the 20 Newsgroups benchmark.
2. Extracts high-dimensional text representations using **TF-IDF vectorization** with strict data leakage prevention (fitted exclusively on the training set).
3. Trains and benchmarks four classic classifiers under identical conditions:
   - **Decision Tree** (Gini criterion)
   - **K-Nearest Neighbors (KNN)** ($k=5$, Minkowski metric)
   - **Support Vector Machine (SVM)** (Linear kernel)
   - **Naïve Bayes** (Multinomial with Laplace smoothing)
4. Evaluates all models head-to-head using Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.
5. Conducts a **What-If sensitivity experiment** testing the effect of TF-IDF vocabulary capacity (`max_features` $\in [500, 1000, 2500, 5000, 10000, 20000]$).
6. Automatically generates publication-ready visualization charts and records full execution logs.

---

## 📂 Project Structure

```
Algorithms of IR/
├── text_classification_study.py    # Main script — runs the entire pipeline & what-if experiment
├── requirements.txt                # Python package dependencies
├── README.md                       # Documentation, theoretical analysis & run instructions
└── outputs/                        # Automatically created when running the script
    ├── comparative_table.png       # Grouped bar chart comparing Accuracy, Precision, Recall, F1
    ├── confusion_matrices.png      # 2×2 grid of confusion matrix heatmaps
    ├── what_if_experiment.png      # Sensitivity curve (Macro F1 vs. max_features)
    └── results_summary.txt         # Complete console log, reports & summary tables
```

---

## 📊 Dataset: 20 Newsgroups (5 Selected Categories)

We select **5 distinct categories** to ensure clean topic boundaries while keeping the task a realistic multi-class problem:

| Category Key | Topic | Typical Vocabulary / Content |
|---|---|---|
| `sci.med` | Medical sciences | Clinical terms, anatomy, health, diseases |
| `sci.space` | Space exploration | NASA, planets, orbit, rockets, astronomy |
| `rec.sport.baseball` | Sports: Baseball | Teams, scores, players, leagues, pitches |
| `talk.politics.mideast` | Politics: Middle East | International relations, peace talks, treaties |
| `comp.graphics` | Computer Graphics | 3D rendering, image formats, algorithms |

- **Total Documents**: ~4,700
- **Split Ratio**: 80% Train, 20% Test (Stratified)
- **Feature Extraction**: TF-IDF with `max_features=10000`, `sublinear_tf=True`, English stop words removed.

---

## ⚙️ Prerequisites & Setup

### 1. Ensure Python is Installed
Python 3.9+ is recommended. Check your Python installation:
```powershell
python --version
```
*(If Python is not recognized, install Python from [python.org](https://www.python.org/downloads/) and ensure **"Add python.exe to PATH"** is checked during setup).*

### 2. (Recommended) Create and Activate a Virtual Environment
In the `Algorithms of IR` directory:

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**Windows Command Prompt (cmd):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 How to Run the Script

### Option A: Complete Run (Baseline Evaluation + What-If Experiment)
To execute the complete pipeline, train all 4 models, run the what-if experiment, and export all plots and summaries:
```powershell
python text_classification_study.py
```

### Option B: Run Baseline Only (Skip What-If Experiment)
If you want a fast run focusing only on the 4-model comparison at `max_features=10000`:
```powershell
python text_classification_study.py --skip-what-if
```

---

## 🔬 How to Run the What-If Experiment

The **What-If Experiment** investigates:
> *"How does the vocabulary size (`max_features`) of TF-IDF affect the classification performance of each algorithm?"*

### 1. Run Only the What-If Experiment
If you want to skip baseline chart generation and exclusively run the vocabulary sweep:
```powershell
python text_classification_study.py --what-if-only
```

### 2. Run with Custom Feature Ranges
You can pass custom vocabulary sizes to test using the `--what-if-features` flag:
```powershell
python text_classification_study.py --what-if-features 500 1000 5000 10000 25000
```

### 3. Combining Flags
```powershell
# Custom features on what-if only:
python text_classification_study.py --what-if-only --what-if-features 1000 3000 6000 12000
```

---

## 📈 Output Deliverables

When the script finishes, the `outputs/` folder contains:

1. **`comparative_table.png`**:
   - High-resolution grouped bar chart displaying Accuracy, Precision, Recall, and F1-score for each model side-by-side with exact value annotations.
2. **`confusion_matrices.png`**:
   - A $2 \times 2$ grid of annotated heatmaps illustrating where each model succeeds and which categories it confuses.
3. **`what_if_experiment.png`**:
   - Multi-line sensitivity plot mapping `max_features` (log scale) against Macro F1-score, highlighting convergence and degradation points across algorithms.
4. **`results_summary.txt`**:
   - Plaintext record containing dataset statistics, per-class classification reports, execution times, and summary tables.

---

## 🧠 Diagnostic & Algorithmic Analysis

### Why does Linear SVM dominate?
Text data represented via TF-IDF resides in high-dimensional, sparse feature spaces. According to Cover's Theorem, high-dimensional spaces are frequently linearly separable. Linear SVM finds the maximum-margin hyperplane, making it resilient to overfitting even with 10,000+ features.

### Why is Multinomial Naïve Bayes so competitive?
Multinomial NB directly models word occurrence probabilities conditioned on document classes. Despite assuming conditional feature independence (which rarely holds in natural language), Laplace smoothing prevents zero probabilities, and the ranking of posterior probabilities remains robust for topical classification.

### Why does KNN struggle?
KNN relies on geometric distance metrics (Minkowski/Euclidean). In a 10,000-dimensional sparse space, the phenomenon known as the **Curse of Dimensionality** causes the distance between any two documents to converge towards a nearly uniform value, drastically degrading neighbor discriminability. Additionally, prediction time is high ($O(N \cdot D)$).

### Why does Decision Tree underperform?
Standard Decision Trees select orthogonal axis-aligned splits on single features at each node. In sparse text where thousands of words weakly signal a class, greedy single-word splits fail to capture the collective context, leading to deep, high-variance trees prone to overfitting.
