# DGSUI: Disentangled Dynamic Graph Framework for Dual-Loop Human Behavioral Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper: 

> **A Disentangled Dynamic Graph Framework for Modeling Dual-Loop Human Behavioral Dynamics and Mitigating Information Echo Chambers**

## 📖 Overview

As artificial intelligence increasingly dictates information exposure, the algorithmic amplification of homogeneous preferences frequently traps individuals in information echo chambers, precipitating systemic "entropy collapse" within socio-technical systems. 

**DGSUI (Decoupling Global and Salient User Intents)** is a generalized computational framework designed to transcend traditional static graph modeling. By structurally replicating human dual-loop cognition, DGSUI effectively disentangles complex interaction trajectories into mutually independent **steady-state (habitual)** and **transient (exploratory)** semantic subspaces. It acts as an algorithmic "entropy firewall," achieving Pareto-optimal balance between predictive utility (accuracy) and system biodiversity (diversity).

### ✨ Core Innovations
1. **Continuous Temporal Awareness:** Replaces discrete time-slicing with a physical time-decay mechanism mapping the psychological Ebbinghaus forgetting curve.
2. **Geometric Orthogonal Disentanglement:** Imposes a strict spatial penalty ($\langle g, s \rangle \approx 0$) to robustly isolate long-term habits from short-term exploratory bursts, preventing semantic collapse.
3. **Intent-Aware Graph Routing:** Integrates Shannon entropy maximization to proactively trace knowledge from topologically distinct high-order neighborhoods.
4. **Adaptive Margin Debiasing:** Dynamically adjusts prediction boundaries based on item popularity to safely mine long-tail knowledge without catastrophic over-fitting.

---

## 🚀 Repository Structure

```text
DGSUI/
├── data/                   # Dataset directory (Taobao, Electronics, Steam)
├── models/
│   ├── dgsui.py            # Main DGSUI architecture
│   └── layers.py           # Core modules (Time-aware MHSA, Orthogonal Engine)
├── utils/
│   ├── data_loader.py      # Chronological sorting and Leave-one-out splitting
│   ├── metrics.py          # R@N, H@N, C@N, B@N evaluation metrics
│   └── loss.py             # Orthogonal, Entropy, and Adaptive Margin BPR losses
├── main.py                 # Training and evaluation loop
├── requirements.txt        # Environment dependencies
└── README.md



⚙️ Installation & RequirementsThe code has been tested under Python 3.8+ and PyTorch 1.12+. To set up the environment, run:Bash# Clone the repository
git clone [https://github.com/YourUsername/DGSUI-Recommendation.git](https://github.com/YourUsername/DGSUI-Recommendation.git)
cd DGSUI-Recommendation

# Create a virtual environment (Recommended)
conda create -n dgsui_env python=3.9
conda activate dgsui_env

# Install dependencies
pip install -r requirements.txt
📊 DatasetsWe evaluate DGSUI on three large-scale, real-world datasets:Taobao: Industrial e-commerce interactions (filtered by 10-core).Amazon Electronics: Highly time-sensitive consumer electronics reviews (filtered by 5-core).Steam: Online gaming platform data featuring frequent intent shifts (filtered by 5-core).Preprocessing:To prevent temporal look-ahead bias, all datasets must be globally sorted by authentic timestamps. We adopt a strict leave-one-out evaluation protocol: the last item for testing, the penultimate for validation, and the rest for training. Place the downloaded raw datasets into the ./data/ directory and run the preprocessing script:Bashpython utils/data_loader.py --dataset taobao
🏃‍♂️ Quick StartTo train and evaluate the DGSUI framework from scratch, use the main.py script. You can specify the dataset and key hyperparameters (e.g., aggregation layers $L$, margin scalar $\gamma$, sequence length $S$).Bash# Example: Running DGSUI on the Taobao dataset
python main.py \
    --dataset taobao \
    --hidden_size 64 \
    --num_layers 3 \
    --seq_len 50 \
    --alpha 0.5 \
    --beta 0.05 \
    --gamma 0.1 \
    --gpu_id 0


