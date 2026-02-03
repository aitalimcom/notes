# Introduction to Bayesian Learning Concepts
*Lecture Notes - January 26, 2026*

## Outline
1.  Maximum Likelihood Estimation (MLE)
2.  Maximum A Posteriori (MAP) Estimation
3.  Bayesian Inference
4.  Bayesian Model Comparison
5.  Bayesian Parameter Estimation
6.  Bayesian Regression
7.  Hierarchical Bayesian Models
8.  Gaussian Processes

---

## 1. Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is a fundamental idea in machine learning used to estimate unknown model parameters using observed data. The central idea of MLE is simple: we choose parameter values that make the observed data most probable according to our model.

### 1.1 Mathematical Expression
The MLE estimate of parameters is defined as:
$$ \hat{\theta}_{MLE} = \arg \max_{\theta} P(D | \theta) $$

Where:
*   $D$ represents the observed dataset
*   $\theta$ represents unknown model parameters
*   $P(D | \theta)$ is the likelihood of data given parameters

### 1.2 Why Likelihood Matters
In machine learning:
*   Data is already observed and fixed.
*   Parameters are unknown and variable.
*   We evaluate how likely the data is for different parameter values.
The parameter values that give the highest likelihood are selected as the MLE estimate.

### 1.3 Log-Likelihood in Practice
The likelihood often involves multiplying many probabilities:
$$ P(D | \theta) = \prod_{i=1}^{n} P(x_i | \theta) $$

Taking the logarithm converts multiplication into addition:
$$ \log P(D | \theta) = \sum_{i=1}^{n} \log P(x_i | \theta) $$
This improves numerical stability and simplifies optimization.

### 1.4 Example: Coin Toss
Consider 10 coin tosses with 7 heads and 3 tails.
Let $p$ be the probability of getting a head.
$$ P(D | p) = p^7 (1 - p)^3 $$
Maximizing this likelihood leads to $\hat{p} = 0.7$. Thus, the estimated probability of heads is the observed fraction of heads.

### 1.5 MLE in Machine Learning
MLE forms the basis of many learning algorithms:
*   Linear regression (least squares)
*   Logistic regression
*   Neural networks (cross-entropy loss)
*   Gaussian models

---

## 2. Maximum A Posteriori (MAP) Estimation

MAP estimation extends MLE by incorporating prior knowledge about parameters. It balances what we observe in data with what we already believe.

### 2.1 Mathematical Expression
MAP estimation is defined as:
$$ \hat{\theta}_{MAP} = \arg \max_{\theta} P(\theta | D) $$

Using Bayes’ theorem:
$$ P(\theta | D) \propto P(D | \theta)P(\theta) $$

### 2.2 Interpretation
MAP estimation considers:
*   **Likelihood:** How well parameters explain the data.
*   **Prior:** Our belief about parameters before seeing data.
The result is a more stable estimate, especially with limited data.

### 2.3 Example: Coin Toss
Suppose we believe coins are usually fair (prior).
If only two tosses give heads:
*   MLE estimates $p = 1$.
*   MAP pulls the estimate closer to $p = 0.5$.
MAP avoids extreme conclusions from small datasets.

### 2.4 MAP and Regularization
In machine learning:
*   L2 regularization corresponds to a Gaussian prior.
*   L1 regularization corresponds to a Laplace prior.
Thus, regularization can be interpreted as MAP estimation.

---

## 3. Bayesian Inference Overview

Bayesian inference treats model parameters as random variables rather than fixed values. Learning becomes a process of updating beliefs using observed data.

### 3.1 Bayes’ Theorem
Bayesian inference is based on:
$$ P(\theta | D) = \frac{P(D | \theta)P(\theta)}{P(D)} $$

Where:
*   $P(\theta)$ is the **Prior**.
*   $P(D | \theta)$ is the **Likelihood**.
*   $P(\theta | D)$ is the **Posterior**.

### 3.2 Predictive Distribution
Bayesian models predict new data using:
$$ P(x_{new} | D) = \int P(x_{new} | \theta)P(\theta | D) d\theta $$
Predictions include uncertainty, not just single values.

---

## 4. Bayesian Model Comparison

Bayesian model comparison evaluates which model best explains the data. It balances model fit with model complexity.

### Model Evidence
The key quantity is model evidence:
$$ P(D | M) = \int P(D | \theta, M)P(\theta | M) d\theta $$
Models with unnecessary complexity are automatically penalized.

### Example: Model Selection
*   Linear regression vs polynomial regression.
*   Few parameters vs many parameters.
Bayesian comparison prefers simpler models unless complexity significantly improves explanation.

---

## 5. Bayesian Parameter Estimation

Instead of a single estimate, Bayesian methods produce a full distribution over parameters. This allows us to quantify uncertainty in learned parameters.

### Why This Matters
*   Robust decision-making.
*   Better performance with small datasets.
*   Confidence-aware predictions.

---

## 6. Bayesian Regression

Bayesian regression places a probability distribution over regression weights. Predictions include both mean values and uncertainty ranges.

### Regression Example
In house price prediction:
*   **Classical regression** predicts a single price.
*   **Bayesian regression** predicts a range of possible prices.

---

## 7. Hierarchical Bayesian Models

Hierarchical models introduce multiple levels of parameters. They allow information sharing across related groups.

### Example: Student Performance
*   Global difficulty level.
*   School-specific difficulty.
*   Individual student performance.
Such models improve learning when data is sparse.

---

## 8. Gaussian Processes (GP)

Gaussian Processes define probability distributions over functions. They are powerful tools for regression and uncertainty modeling.

### 8.1 GP Representation
A Gaussian Process is written as:
$$ f(x) \sim GP(m(x), k(x, x')) $$
Where $m(x)$ is the mean function and $k(x, x')$ is the kernel.

### 8.2 Applications
*   Time-series forecasting.
*   Bayesian optimization.
*   Spatial data modeling.
*   Robotics and control.

---

## Summary
*   **MLE** fits parameters using data alone.
*   **MAP** combines data with prior knowledge.
*   **Bayesian inference** models uncertainty explicitly.
*   **Advanced Bayesian models** enable robust ML systems.
