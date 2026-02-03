# Lecture Notes: Isolation Forest with Worked Dummy Example

## 1. Introduction
Isolation Forest is an anomaly detection method based on the principle that **anomalies are easier to isolate** in a random binary partitioning process than normal points.
*   **Isolation Tree (iTree):** The structure used for isolation.
*   **Principle:** Shorter average path lengths indicate easier isolation and thus stronger anomaly characteristics.

## 2. Isolation Trees
An Isolation Tree is constructed by:
1.  Selecting a random feature.
2.  Selecting a random split value between the minimum and maximum of that feature.
3.  Partitioning the data recursively until each observation is isolated or a maximum height is reached.

For a point $x$, the path length $h(x)$ in a single tree is the number of edges from the root node to the terminating node in which $x$ becomes isolated.

---

## 3. Dummy Dataset
Consider the following dataset with sample size $n = 5$:

| Point | Feature A | Feature B |
| :--- | :--- | :--- |
| **P1** | 1 | 10 |
| **P2** | 2 | 12 |
| **P3** | 3 | 11 |
| **P4** | 50 | 200 |
| **P5** | 4 | 13 |

*Observation:* By inspection, **P4** appears unusual (values 50, 200) relative to others (values < 5, < 15).

---

## 4. Tree Construction: Tree 1

**Step 1:** Select **Feature A**. Range [1, 50]. Random split = **6**.
*   Left group (A < 6): {P1, P2, P3, P5}
*   Right group (A $\ge$ 6): {P4}
    *   **P4 is isolated.** Path length $h(P4) = 1$.

**Step 2:** Split Left Group. Select **Feature B**. Range [10, 13]. Random split = **11.5**.
*   Left (B < 11.5): {P1 (10), P3 (11)}
*   Right (B $\ge$ 11.5): {P2 (12), P5 (13)}

**Step 3:** Split Left-Left Group ({P1, P3}). Select **Feature A**. Range [1, 3]. Random split = **2**.
*   Left (A < 2): {P1} $\rightarrow$ Isolated. Path $h(P1) = 3$.
*   Right (A \ge 2): {P3} $\rightarrow$ Isolated. Path $h(P3) = 3$.

**Step 4:** Split Left-Right Group ({P2, P5}). Select **Feature A**. Range [2, 4]. Random split = **3**.
*   Left (A < 3): {P2} $\rightarrow$ Isolated. Path $h(P2) = 3$.
*   Right (A \ge 3): {P5} $\rightarrow$ Isolated. Path $h(P5) = 3$.

**Path Lengths (Tree 1):**
*   $h(P4) = 1$
*   $h(P1, P2, P3, P5) = 3$

---

## 5. Tree 2 & Tree 3 (Simulation)
*Assuming random splits consistently isolate P4 early due to its extreme values.*

**Tree 2 Construction:**
1.  Feature B, Range [10, 200], Split = 50.
    *   Right: {P4} (Isolated, $h=1$).
    *   Left: {P1, P2, P3, P5}.
2.  Recursive splits isolate others at depth 3.

**Tree 3 Construction:**
1.  Feature A, Split = 30.
    *   Right: {P4} (Isolated, $h=1$).
    *   Left: {P1, P2, P3, P5}.
2.  Recursive splits isolate others at depth 3.

**Average Path Length $E(h(x))$:**
Since all three trees yield identical values:
*   $E(h(P4)) = 1$
*   $E(h(\text{others})) = 3$

---

## 6. Computing Anomaly Scores

The anomaly score $s(x, n)$ is defined as:
$$ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} $$

### 6.1 Normalization Factor $c(n)$
The average path length of unsuccessful search in a BST, given by:
$$ c(n) = 2H(n - 1) - \frac{2(n - 1)}{n} $$
Where $H(i)$ is the harmonic number $H(i) \approx \ln(i) + 0.577$.
For small $n=5$, we sum directly: $H(4) = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} = 2.0833$.

$$ c(5) = 2(2.0833) - \frac{2(4)}{5} $$
$$ c(5) = 4.1666 - 1.6 = 2.5666 \approx 2.57 $$

### 6.2 Score Calculation
**For Anomaly P4:**
$$ s(P4, 5) = 2^{-\frac{1}{2.57}} \approx 2^{-0.389} \approx \mathbf{0.76} $$

**For Normal Point P1:**
$$ s(P1, 5) = 2^{-\frac{3}{2.57}} \approx 2^{-1.167} \approx \mathbf{0.44} $$

---

## 7. Conclusion & Interpretation
| Point | $E(h)$ | Score $s$ | Interpretation |
| :--- | :--- | :--- | :--- |
| **P4** | 1 | **0.76** | **High Anomaly** (Score close to 1) |
| **P1..P5** | 3 | **0.44** | **Normal** (Score < 0.5) |

*   **$s \approx 1$:** Strong anomaly.
*   **$s < 0.5$:** Normal instance.
*   **$s \approx 0.5$:** No distinct anomaly.

This confirms that Isolation Forest successfully flagged **P4** as an anomaly purely based on how few splits were needed to isolate it.
