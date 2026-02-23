# Spatio-Temporal Graph Data Generation for Network Intrusion Detection

This repository contains robust, highly-optimized data generation scripts for transforming raw network traffic datasets (CSVs) into Spatio-Temporal Graph formats and flattened arrays. 

These scripts are specifically designed to evaluate Graph Neural Networks (GNNs) and Spatio-Temporal models for Network Intrusion Detection Systems (NIDS). They strictly adhere to chronological causality, explicitly preventing the severe "temporal data leakage" issues prevalent in existing NIDS literature.

## 🚀 Key Features

* **Strict Chronological & Stratified Splitting**: Avoids naive `train_test_split` on global data. Data is grouped by time windows (snapshots), and splits are performed chronologically or via snapshot-level stratified sampling to maintain causal realism.
* **Authentic Topological Preservation**: Unlike some baselines that randomize IPs to prevent overfitting, this codebase hashes real Source and Destination IPs to construct genuine, dynamic interaction graphs.
* **Unified Output Formats**: Simultaneously generates PyTorch Geometric (`.pt`) graph sequences for Spatial-Temporal GNNs, and flattened arrays (`.npz`) for traditional Machine Learning/Sequence models (e.g., Random Forest, GRU, MLP).
* **Class Imbalance & Fallback Handling**: Includes robust mechanisms to ensure validation sets contain sufficient anomaly/attack edges, automatically adjusting split points (e.g., in ISCX-IDS-2012) or performing adaptive block searching (e.g., in Darknet2020) if necessary.

## 📂 Supported Datasets

The repository provides tailored generation scripts for four major NIDS/Traffic Classification datasets:

| Dataset | Script | Output Directory |
| :--- | :--- | :--- |
| **CIC-IDS2017** | `generate_ids2017_dataset.py` | `processed_data/cic_ids2017/` |
| **CIC-Darknet2020** | `generate_2020_dataset.py` | `processed_data/darknet2020_block/` |
| **UNSW-NB15** | `generate_nb15_dataset.py` | `processed_data/unsw_nb15/` |
| **ISCX-IDS-2012** | `generate_iscx2012_dataset.py` | `processed_data/iscx_ids2012/` |

## 🛠️ Environment Requirements

Ensure you have the following packages installed:

```bash
pip install pandas numpy scikit-learn tqdm joblib
pip install torch torchvision torchaudio
pip install torch_geometric
