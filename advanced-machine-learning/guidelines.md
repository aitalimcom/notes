# 🎓 Student Guidelines & Troubleshooting

Welcome to the **Advanced Machine Learning (AICC 303)** course repository. This document provides essential guidelines for setting up your environment, running the notebooks, and troubleshooting common issues.

## 🛠️ Environment Setup

### 1. Recommended Setup (Anaconda/Miniconda)
We recommend using **Anaconda** or **Miniconda** to manage your Python environment.

```bash
# Create a new environment
conda create -n aml_course python=3.10
conda activate aml_course

# Install core libraries
pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn
```

### 2. Installing Deep Learning Frameworks
This course uses **PyTorch** for all deep learning tasks.

**Check if you have a GPU (Optional but Recommended):**
Running Deep RL and NLP models on a CPU can be slow. If you have an NVIDIA GPU:

```bash
# Install PyTorch with CUDA support (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**
```bash
pip install torch torchvision
```

### 3. Installing Other Dependencies

```bash
# Time Series
pip install statsmodels

# NLP
pip install nltk transformers

# Reinforcement Learning (Important: Use gymnasium)
pip install gymnasium
```

---

## ⚠️ Common Issues & Solutions

### 1. `AttributeError: module 'numpy' has no attribute 'bool8'`

**Context:** This error occurs in the **Reinforcement Learning (Unit 5)** notebooks.
**Cause:** The older `gym` library is incompatible with newer versions of `numpy` (2.0+).

**Solution:**
We have migrated the notebooks to use `gymnasium`, the modern replacement for `gym`. Ensure you install it:
```bash
pip install gymnasium
```
If you still see `import gym` in any legacy code or external resources, replace it with:
```python
import gymnasium as gym
```

### 2. `LookupError: Resource punkt_tab not found`

**Context:** This error occurs in **NLP (Unit 4)** notebooks during tokenization.
**Cause:** Newer versions of NLTK require a specific tokenizer table (`punkt_tab`) that isn't included in the standard `punkt` download.

**Solution:**
Run the following code in a notebook cell to download all necessary data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # Critical for newer NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 3. `Kernel Died` or `Out of Memory`

**Context:** Deep Learning notebooks (Unit 2.4, 3.1, 4.2, 5.5).
**Cause:** Your computer ran out of RAM or VRAM (GPU memory).

**Solutions:**
*   **Reduce Batch Size:** In training loops, change `batch_size=32` to `16` or `8`.
*   **Reduce Sequence Length:** In Time Series/NLP, shorten the input sequences.
*   **Restart Kernel:** Jupyter kernels sometimes accumulate memory. Restart via `Kernel -> Restart`.

---

## 📚 Notebook Guidelines

1.  **Sequential Order:** Run cells in order (Top to Bottom). Skipping cells (especially imports or variable definitions) will cause errors.
2.  **Interactive Visualizations:** Some notebooks use interactive plots. Ensure you trust the notebook if prompted by Jupyter.
3.  **Assignments:**
    *   Assignments are located at the end of each Unit folder (e.g., `2.5_assignment.ipynb`).
    *   Attempt the questions yourself before checking solutions or running the provided answer cells.
    *   Assignments are designed to test both theoretical understanding and coding skills.

## 🤝 Need Help?

*   Check the official documentation for [PyTorch](https://pytorch.org/docs/), [Scikit-Learn](https://scikit-learn.org/stable/), or [Gymnasium](https://gymnasium.farama.org/).
*   Refer to the Reference Books listed in the syllabus.
