# Advance Machine Learning Course Implementation Plan

This plan outlines the creation of educational materials for the "Advanced Machine Learning" course (AICC 303), mapping the syllabus and provided references to practical Jupyter notebooks.

## Phase 1: Unit 1 - Probabilistic Models and Inference (Completed)
- [x] **1.1 Probability Distributions**: Created `1.1_probability_distributions.ipynb`, `1.2_bernoulli_binomial.ipynb`, `1.3_multinomial_dirichlet.ipynb`.
- [x] **Practice**: Created `chapter 1 assignment.ipynb` and `chapter 1 assignment_SOLVED.ipynb`.

## Phase 2: Unit 2 - Time Series Analysis (In Progress)
The goal is to bridge classical forecasting methods (from Hyndman's book) with modern Deep Learning approaches.

- [x] **2.1 Classical Time Series Analysis** (`2.1_classical_time_series_analysis.ipynb`)
    - Covers: Introduction, Decomposition, Moving Averages, ACF/PACF, Stationarity, ARIMA.
    - Status: **Created**.
- [ ] **2.2 Deep Learning for Time Series**
    - **Objective**: Cover Syllabus topic 2.5 (RNN, LSTM, GRU).
    - **Content**:
        - Data preparation for sequence models (Sliding Windows).
        - Vanishing Gradient problem context.
        - Implementing Simple RNN using PyTorch/TensorFlow.
        - Implementing LSTM and GRU.
        - Comparison: ARIMA vs LSTM on the same dataset.
- [ ] **2.3 Unit 2 Assignment**
    - Practice problems covering both ARIMA (Classical) and LSTM (Deep Learning) implementations.

## Phase 3: Unit 3 - Anomaly Detection
- [ ] **3.1 Anomaly Detection Methods**
    - **Objective**: Cover Syllabus Unit 3.
    - **Content**:
        - Statistical Methods (Z-score, IQR).
        - Isolation Forest (Syllabus 3.2).
        - One-class SVM (Syllabus 3.4).
        - Autoencoders for Anomaly Detection (Syllabus 3.3) - *Deep Learning approach*.

## Phase 4: Unit 4 - NLP & Neural Attention
- [ ] **4.1 NLP Fundamentals & Embeddings**
    - Tokenization, Stemming/Lemmatization.
    - Word Embeddings (Word2Vec, GloVe).
- [ ] **4.2 Attention & Transformers**
    - Attention Mechanisms (Dot-product, Scaled).
    - The Transformer Architecture (Self-Attention).
    - Fine-tuning a pre-trained model (e.g., BERT/GPT) for a downstream task.

## Phase 5: Unit 5 - Reinforcement Learning
- [ ] **5.1 RL Fundamentals & Tabular Methods**
    - MDPs, Bellman Equations.
    - Q-Learning, SARSA.
- [ ] **5.2 Deep Reinforcement Learning**
    - Deep Q-Networks (DQN).
    - Policy Gradients (REINFORCE).

## Execution Strategy
1.  Complete **Unit 2** (Deep Learning & Assignment).
2.  Proceed systematically through Units 3, 4, and 5.
3.  Ensure each unit has both a **Theoretical/Tutorial Notebook** and a **Practice Assignment**.
