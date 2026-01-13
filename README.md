# Phishing & Malicious URL Detection using Deep Learning

## 📌 Project Overview

This repository presents a **comprehensive, multi-dataset study on phishing and malicious URL detection** using deep learning models. The project focuses on **robust preprocessing, cross-dataset generalization, and systematic experimental evaluation** to understand how different datasets and training strategies affect performance.

Unlike many prior works that train on a single dataset, this work evaluates **model stability and transferability across heterogeneous phishing datasets**.

---

## ✨ Research Novelty & Key Contributions

1. **Multi-source Dataset Integration**

   * Combines **4 widely used phishing/malicious URL datasets** (Kaggle + HuggingFace).
   * Ensures consistent schema (`url`, `label`) and unified label encoding.

2. **Strict Data Hygiene**

   * Duplicate URL removal across datasets
   * Label normalization and binary encoding
   * Stratified train/validation/test splits

3. **Cross-Dataset Generalization Study**

   * Models trained on one dataset and evaluated on others
   * Highlights real-world robustness beyond dataset bias

4. **Stage-wise Training Diagnostics**

   * Tracks training accuracy, validation accuracy, loss curves, and convergence behavior

5. **Reproducible & Modular Pipeline**

   * Fully automated preprocessing and lazy dataset loading
   * Clear separation between preprocessing, training, and evaluation

---

## 📊 Datasets Used

| Dataset ID | Source                  | Description                      |
| ---------- | ----------------------- | -------------------------------- |
| Dataset 1  | Kaggle (Malicious URLs) | Mixed malicious and benign URLs  |
| Dataset 2  | Kaggle (PhiUSIIL)       | Phishing-focused URLs            |
| Dataset 3  | HuggingFace (kmack)     | Large-scale phishing URL corpus  |
| Dataset 4  | Kaggle (Tarun Tiwari)   | Real-world phishing website URLs |

All datasets are standardized to:

```text
url   → string
label → 1 (phishing/malicious), 0 (benign)
```

---

## 🧪 Training Pipeline (Stage-wise)

### **Stage 1: Data Preprocessing** (`S01_dataset_preprocessing_pipeline.py`)

* Dataset download (Kaggle + HuggingFace)
* Column normalization
* Label filtering & encoding
* Duplicate URL removal
* Stratified splitting (Train / Validation / Test)

📈 **Graphs Produced**:

* Label distribution per split
* Dataset size comparison

---

### **Stage 2: Exploratory Data Analysis** (`S02_exploratory_dataset_analysis.ipynb`)

* Class imbalance analysis
* URL length distribution
* Phishing vs benign structural patterns

📈 **Graphs Produced**:

* Bar plots for class distribution
* Histograms for URL length

---

### **Stage 3: Model Training (Per Dataset)** (`S03_multi_dataset_training_experiments.ipynb`)

* CNN-based deep learning model
* Dataset-specific training
* Early stopping & checkpointing

📈 **Graphs Produced**:

* Training Accuracy vs Epochs
* Validation Accuracy vs Epochs
* Training Loss vs Epochs

---

### **Stage 4: Cross-Dataset Evaluation & Comparison** (`S04_model_performance_comparison.ipynb`)

* Compare models trained on different datasets
* Evaluate generalization performance

📈 **Graphs Produced**:

* Test Accuracy (Bar Chart)
* Train vs Validation Accuracy Comparison
* Accuracy Degradation Across Datasets

---

## 📈 Experimental Results

### **TABLE I. Performance Comparison of the Proposed Model Across Four Phishing URL Datasets**

| Dataset                              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------------------------ | -------- | --------- | ------ | -------- | ------- |
| Dataset 1: Malicious URLs            | 0.9848   | 0.9824    | 0.9322 | 0.9566   | 0.9969  |
| Dataset 2: PhiUSIIL Phishing         | 0.9978   | 0.9967    | 0.9994 | 0.9981   | 0.9986  |
| Dataset 3: KMack Phishing URLs       | 0.9170   | 0.9094    | 0.9262 | 0.9177   | 0.9739  |
| Dataset 4: Kaggle Phishing Site URLs | 0.9817   | 0.9768    | 0.9410 | 0.9586   | 0.9966  |

---

### **TABLE II. Performance Comparison of Ablation Models on Phishing URL Detection**

| Model                          | Accuracy   | Precision  | Recall     | F1-score   | ROC-AUC    | R²     |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ------ |
| MLP on URL Embeddings          | 0.9518     | 0.8974     | 0.8878     | 0.8926     | 0.9781     | 0.7491 |
| Residual Multi-Kernel CNN      | 0.9778     | 0.9742     | 0.9262     | 0.9496     | 0.9945     | 0.8947 |
| Temporal BiLSTM                | 0.9796     | 0.9750     | 0.9332     | 0.9536     | 0.9956     | 0.9068 |
| CNN + BiLSTM Hybrid            | 0.9797     | 0.9786     | 0.9300     | 0.9537     | 0.9959     | 0.9083 |
| SE + Attention Pooling         | 0.9795     | 0.9671     | 0.9412     | 0.9540     | 0.9957     | 0.9074 |
| DAP + CNN + BiLSTM             | 0.9821     | 0.9743     | 0.9456     | 0.9597     | 0.9966     | 0.9208 |
| **Proposed Best Hybrid Model** | **0.9829** | **0.9717** | **0.9518** | **0.9616** | **0.9970** | —      |

---

## 🧠 Key Observations

* Validation accuracy plateaus earlier for smaller datasets
* Dataset bias significantly impacts cross-dataset testing
* CNN-based URL models benefit from diverse training data

---

## 🔮 Future Work

* Transformer-based URL encoders
* Federated learning across datasets
* Adversarial URL robustness testing
* Real-time deployment pipeline

---

## 📜 Citation (Suggested)

If you use this work in your research, please cite:

> *Multi-Dataset Deep Learning Framework for Phishing URL Detection*, 2026

---

## 👤 Author

**Ashutosh Ranjan**
Faculty of Technology, University of Delhi

---

✅ *This repository is designed to be reproducible, extensible, and research-review ready.*
