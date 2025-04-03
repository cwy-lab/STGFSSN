# Spatio-temporal Graph with Frequency State Space Network

This project implements a **Graph-Mamba** model for **multi-sensor time series forecasting**. The model integrates **graph neural networks, cross-attention mechanisms, and Mamba state-space modeling**, designed to capture **both temporal and cross-sensor dependencies** in the data. It is specifically built for **multi-sensor prediction tasks** using **PyTorch**.

## 🚀 Features  

- **Graph Attention Networks (GAT)**: Uses multi-head **cross-attention** to capture relationships between different sensors.  
- **Mamba State Space Modeling**: Incorporates **FFT-based state-space representation** to enhance long-term dependencies.  
- **Multi-Branch Architecture**:  
  - **Graph Attention Branch** → Extracts inter-sensor dependencies.  
  - **FFT-Based State-Space Branch** → Captures frequency domain information.  
  - **CNN Skip Connection Branch** → Preserves raw time-domain features.  
- **Min-Max Scaling & Inverse Scaling**: Data is **normalized before training** and **rescaled** after prediction.  
- **Modular Codebase**: The project is structured into **separate files for the dataset, model, and main script**, making it easy to extend and modify.

---

## 📥 Installation  

To set up the environment, run:

```bash
git clone https://github.com/cwy94/STGFSSN.git
cd STGFSSN
pip install -r requirements.txt
```
---

## 🔧 Usage
To train and test the model, simply run:
```bash
python main.py
```
