# 🧠 Deep Learning & Neural Networks with PyTorch

This repository tracks hands-on deep learning experiments built with **PyTorch**, starting from single-neuron linear computations and tensor conversions up to artificial neural network workflows.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`)
* **Numerical & Data Handling:** NumPy, Pandas
* **Environment:** Jupyter Notebook

---

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **`neurons.ipynb`** | Single-neuron linear layer (`nn.Linear`) setup with manual seed initialization, forward pass, and verification using `weight @ inputs + bias` |
| **`project_dl1/`** | Sub-project directory containing neural network training experiments and datasets |
| **`project_dl1/ANN.ipynb`** | Artificial Neural Network workflow including scaled NumPy-to-PyTorch `float32` tensor conversion for train and test sets |
| **`project_dl1/powerplant_data.csv`** | Dataset used for neural network experimentation |
| **`.ipynb_checkpoints/`** | Jupyter Notebook autosave checkpoint directories |

---

## 📌 Key Concepts Covered

* **Single Neuron & Linear Layer (`nn.Linear`):** Defining input/output features (`in_features`, `out_features`), running forward passes, and verifying linear outputs mathematically via `weights @ inputs + bias`.
* **Reproducibility:** Setting random seeds with `torch.manual_seed(...)` for consistent tensor and parameter initialization.
* **Tensor Preparation:** Converting scaled NumPy arrays (`X_train_scaled`, `X_test_scaled`) and target values (`y_train.values`) into `torch.float32` tensors with reshaped dimensions (`.view(-1, 1)`).

