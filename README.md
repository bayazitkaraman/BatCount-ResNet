# BatCount-ResNet

This repository provides the training and benchmarking code for Leveraging Pretrained ResNet Architectures for Enhanced Real-Time Bat Counting. We evaluate ResNet-18/34/50/101 with 5-fold cross-validation on a curated bat emergence ROI dataset and report accuracy, per-class metrics, confusion matrices, ROC/PR curves, and end-to-end throughput. The code emphasizes reproducibility (fixed seeds, pinned dependencies, run manifest) and practical deployment (per-fold checkpoints, latency measurements, simple CLI/config). ResNet-18 offers the best accuracy–throughput trade-off, while ResNet-34 attains the highest mean accuracy.

Code for the paper: *Leveraging Pretrained ResNet Architectures for Enhanced Real-Time Bat Counting*.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/bayazitkaraman/BatCount-ResNet.git
cd BatCount-ResNet

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Then run the main script:
python ResnetBatsCounting.py
