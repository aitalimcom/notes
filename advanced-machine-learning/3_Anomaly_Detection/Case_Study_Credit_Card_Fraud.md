# Case Study: Credit Card Fraud Detection

*Based on popular Kaggle Kernels by Gabriel Preda and Janio Martinez Bachmann*

## 1. Introduction
Credit Card Fraud Detection is a classic **Anomaly Detection** problem where the goal is to identify fraudulent transactions among a vast majority of legitimate ones.

## 2. The Challenge: Imbalanced Datasets
Real-world fraud datasets are highly imbalanced (e.g., 0.17% fraud vs 99.83% legitimate).
*   **Problem:** A naive model can achieve 99.83% accuracy by simply predicting "Legitimate" for every transaction, but it will fail to detect any fraud.
*   **Solution Strategy:** We cannot rely on accuracy. We must focus on how we handle the imbalance and what metrics we use.

---

## 3. Strategies for Dealing with Imbalance

### 3.1 Resampling Techniques
*   **Undersampling:** Randomly removing examples from the majority class (Legitimate) to match the count of the minority class (Fraud).
    *   *Pros:* Fast training.
    *   *Cons:* Loss of potentially valuable information.
*   **Oversampling (SMOTE):** Synthetic Minority Over-sampling Technique. Creating synthetic points for the minority class based on nearest neighbors.
    *   *Pros:* Retains information.
    *   *Cons:* Can lead to overfitting; computationally expensive.

### 3.2 Class Weights
Assigning a higher penalty cost for misclassifying the minority class during model training.
*   Most libraries (Scikit-Learn, XGBoost) allow calculating `scale_pos_weight` or `class_weight`.

---

## 4. Modeling Approaches

### 4.1 Supervised Learning (Predictive Models)
When labels are available, robust classifiers are often used.
*   **Logistic Regression:** Good baseline.
*   **Random Forest:** Handles non-linearities well.
*   **Gradient Boosting (XGBoost / LightGBM):** Often the state-of-the-art for tabular data.
    *   *Key:* Tuning hyperparameters like `max_depth`, `learning_rate`, and `scale_pos_weight`.

### 4.2 Anomaly Detection (Unsupervised/Semi-Supervised)
When labels are scarce or we want to detect "novel" fraud patterns.
*   **Isolation Forest:** Isolates anomalies by randomly partitioning data. Anomalies require fewer splits to be isolated.
*   **Autoencoders:** Train a neural network to reconstruct legitimate transactions. High reconstruction error implies fraud (anomaly).
*   **One-Class SVM:** Fits a hyperplane that best separates the normal data from the origin (or surrounds it).

---

## 5. Evaluation Metrics

**STOP using Accuracy.**

### 5.1 Confusion Matrix
*   **True Positives (TP):** Fraud correctly identified as Fraud.
*   **False Negatives (FN):** Fraud missed (Predicting Legitimate). **Critical Cost**.
*   **False Positives (FP):** Legitimate flagged as Fraud (False Alarm). Customer annoyance.

### 5.2 Key Metrics
*   **Recall (Sensitivity):** $ TP / (TP + FN) $. How many frauds did we catch? (Maximize this).
*   **Precision:** $ TP / (TP + FP) $. Of the ones we flagged, how many were actually fraud?
*   **F1-Score:** Harmonic mean of Precision and Recall.
*   **AUPRC (Area Under Precision-Recall Curve):** The gold standard metric for highly imbalanced datasets. Better than ROC-AUC in this context.

---

## 6. Dimensionality Reduction & Visualization
*   **PCA (Principal Component Analysis):** Reduces dimensions while preserving variance.
*   **t-SNE:** Non-linear reduction effective for visualization. Can reveal if fraud cases form a distinct "cluster" separated from normal transactions.
