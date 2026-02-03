# Local Outlier Factor (LOF)
*Based on the paper: "LOF: Identifying Density-Based Local Outliers"*

## 1. Motivation: Local vs. Global Outliers
Traditional anomaly detection methods (like distance-based outliers) often take a **global view**. They assume that outliers are points "far away" from the rest of the data.
*   **The Problem:** In datasets with clusters of varying densities, a global threshold fails. A point might be far from a dense cluster (and thus an outlier relative to it) but the distance might be "normal" for a sparse cluster.
*   **Solution:** **Local Outlier Factor (LOF)** acts on the principle that being an outlier is not a binary property but a degree. It compares the local density of an object to the local densities of its neighbors.

---

## 2. Formal Definitions

### 2.1 k-distance(p)
The distance from point $p$ to its $k$-th nearest neighbor.

### 2.2 Reachability Distance
Measures how "reachable" point $p$ is from point $o$.
$$ \text{reach-dist}_k(p, o) = \max \{ \text{k-distance}(o), d(p, o) \} $$
*   If $p$ is close to $o$, the distance is replaced by the $k$-distance of $o$ (smoothing effect).
*   If $p$ is far, it uses the actual distance.

### 2.3 Local Reachability Density (lrd)
Inverse of the average reachability distance from $p$'s neighbors.
$$ \text{lrd}(p) = \frac{1}{ \frac{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}{|N_k(p)|} } $$
*   Intuitively: How dense is the neighborhood around $p$?

### 2.4 Local Outlier Factor (LOF)
The ratio of the average local density of $p$'s neighbors to the local density of $p$ itself.
$$ \text{LOF}(p) = \frac{ \sum_{o \in N_k(p)} \frac{\text{lrd}(o)}{\text{lrd}(p)} }{ |N_k(p)| } $$

---

## 3. Interpretation of LOF Values
*   **LOF $\approx$ 1:** The point is in a cluster (similar density to neighbors).
*   **LOF > 1:** The point is an outlier (lower density than neighbors).
*   **LOF < 1:** The point is in a denser region than its neighbors (rare).

**Key Insight:** LOF captures the *relative* degree of isolation.

---

## 4. Parameter Selection: MinPts
The Algorithm depends heavily on $k$ (MinPts). The paper suggests using a **range** of MinPts values rather than a single fixed value.
*   **Lower Bound:** ~10 (to remove statistical fluctuations).
*   **Upper Bound:** Max size of a "local" cluster.
*   **Heuristic:** Compute LOF for a range (e.g., 10 to 50) and take the **maximum** LOF score for each point.

---

## 5. Case Studies Comparison
1.  **Synthetic Data:** LOF successfully identifies outliers near dense clusters that distance-based methods miss.
2.  **Hockey Data:** Identified players like Vladimir Konstantinov (high points, high penalties) as outliers.
3.  **Soccer Data:** Identified players like Michael Preetz (top scorer) and Goalies who score goals (Hans-Jörg Butt) as local outliers relative to their position groups.

## 6. Pros and Cons
*   **Pros:** Handles variable density clusters effectively. Provides a continuous "outlier score".
*   **Cons:** Computationally expensive ($O(n^2)$ without indexing). selecting the MinPts range can be tricky.
