# Parkinson’s Disease Detection from EEG using GCN & MCPNet

This repository contains two independent deep-learning pipelines for detecting **Parkinson’s Disease (PD)** from **EEG brain signals**:

• A **Graph Convolutional Network (GCN)** that models brain connectivity as a graph  
• A **MCPNet (CNN-based)** model that learns spatial-frequency patterns from EEG  

The goal of this project is to study how different neural architectures perform on the same neurological disorder using EEG-derived features.

---

## 🧬 Why EEG for Parkinson’s?

Parkinson’s disease alters neural firing and synchronization patterns across different brain regions. EEG provides a **non-invasive way** to observe these changes, but the data is:

- Noisy  
- High-dimensional  
- Highly subject-dependent  

This project uses **signal processing, functional connectivity, and deep learning** to extract useful patterns from EEG.

---

## 🧠 Project Structure
```
parkinsons-eeg-gnn/
│
├── gcn/        → Graph Neural Network pipeline
│   ├── train.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── process_features.py
│   ├── verify_features.py
│   ├── check_channels.py
│   ├── best_gcn_model.pth
│   └── labels.csv
│
├── mcpnet/     → CNN-based MCPNet pipeline
│   ├── mcnet_model.py
│   ├── train_mcnet_uc.py
│   ├── train_mcnet_iowa.py
│   ├── train_test_common.py
│   ├── dataset_loader.py
│   ├── epoch_uc.py
│   ├── preprocess_uc.py
│   ├── extract_features.py
│   └── ...
│
└── README.md
```


Large EEG data, extracted features, and intermediate files are intentionally **not uploaded** to GitHub.

---

## ⚙️ EEG Processing Pipeline

Both models rely on the same signal-processing pipeline:

1. Band-pass filtering  
2. Artifact removal (eye blinks, noise)  
3. Epoch segmentation  
4. Feature extraction (PSD, band power, connectivity)  

For GCN, these are converted into **brain graphs**.  
For MCPNet, they are converted into **CNN-compatible tensors**.

---

## 🕸️ GCN Model

The GCN treats EEG channels as **nodes** and functional connectivity as **edges**.

This allows the model to learn:
- Which brain regions interact abnormally
- Network-level Parkinson’s patterns

The adjacency matrix is derived from EEG connectivity measures such as coherence or correlation.

---

## 🧩 MCPNet Model

MCPNet is a **CNN-based EEG classifier** that learns:
- Spatial patterns across electrodes  
- Frequency-domain features  
- Temporal variations across epochs  

This provides a strong baseline against which the GCN is compared.

---

## 📊 Results

The current models achieve approximately:

**~62% classification accuracy**

This reflects the difficulty of Parkinson’s detection from EEG due to:
- Small datasets  
- High inter-subject variability  
- Weak surface EEG biomarkers  

The focus of this project is **methodological correctness and extensibility**, not overfitting.

---

## 🚀 Future Improvements

- More advanced GNNs (GAT, Graph Transformers)  
- Cross-subject normalization  
- Larger EEG datasets  
- End-to-end learning instead of hand-crafted features  

---

## 🧪 How to Run

Install dependencies:
```bash
pip install numpy scipy torch mne
python gcn/train.py
python mcpnet/train_mcnet_uc.py
