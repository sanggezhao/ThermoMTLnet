# ThermoMTLnet

ThermoMTLnet is a research project exploring multi-task feedforward neural networks for molecular thermodynamic property prediction.  
The model integrates descriptor-based features with physics-informed constraints to improve consistency and generalization under small-sample scenarios.

> **Note:**  
> This repository provides a minimal code structure for reference only.  
> It is *not* intended as a full reproduction package for the associated manuscript.

---

## 🔧 Features

- Multi-task learning architecture for predicting multiple thermodynamic properties.
- Modular data preprocessing and feature engineering pipeline.
- Physics-guided consistency loss (PINN-style constraints).
- Built using **PyTorch** and **PyTorch Lightning**.
- Supports molecular descriptor inputs (RDKit / Mordred / custom features).

---

## 📁 Repository Structure

ThermoMTLnet/
│
├── data/                 # (Optional) Placeholder directory for datasets
├── models/               # Model definitions (FNN, MultiTask, PINN-loss, etc.)
├── scripts/              # Training, evaluation, and preprocessing scripts
├── utils/                # Helper functions (metrics, plotting, loaders)
│
├── environment.yml       # Minimal environment specification
└── README.md             # Project description (this file)

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/sanggezhao/ThermoMTLnet
cd ThermoMTLnet

	2.	Create the conda environment:

conda env create -f environment.yml
conda activate ThermoMTLnet


⸻

🚀 Quick Start (Minimal Example)

This is a schematic example showing the workflow.
It does not include full training code or datasets.

python scripts/train.py \
    --config configs/default.yaml \
    --output results/


⸻

📦 Dependencies

Key packages include:
	•	Python ≥ 3.9
	•	PyTorch
	•	PyTorch Lightning
	•	RDKit
	•	NumPy / Pandas
	•	Scikit-learn

A complete environment file is provided as environment.yml.

⸻

📄 License

This project is released for academic reference.
Please contact the author if you intend to use the code for other purposes.

⸻

📬 Contact

For questions regarding this repository:

Author: Sang Gezhao
Email: sanggz@sari.ac.cn
GitHub: https://github.com/sanggezhao

⸻

Thank you for your interest in ThermoMTLnet!

---
