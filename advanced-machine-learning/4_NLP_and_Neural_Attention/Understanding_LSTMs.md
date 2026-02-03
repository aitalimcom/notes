# Understanding LSTM Networks
*Based on Christopher Olah's Blog Post*

## 1. Recurrent Neural Networks (RNN)
Traditional feedforward neural networks assume all inputs (and outputs) are independent of each other. This is a problem for sequential data like language. RNNs address this by having loops, allowing information to persist.

*   An RNN can be thought of as multiple copies of the same network, each passing a message to a successor.
*   **Problem:** RNNs struggle with **Long-Term Dependencies**. As the gap between relevant information and the point where it is needed grows, RNNs become unable to learn to connect the information due to vanishing gradients.

---

## 2. LSTM Networks: The Core Idea
Long Short Term Memory networks (LSTMs) are explicitly designed to avoid the long-term dependency problem.

### 2.1 The Cell State
The key to LSTMs is the **cell state** ($C_t$). It acts like a conveyor belt, running straight down the entire chain with only minor linear interactions. It is very easy for information to flow along it unchanged.

### 2.2 Gates
LSTMs use **gates** to regulate the flow of information. Gates are composed of a sigmoid neural net layer and a pointwise multiplication operation.
*   Sigmoid output $\in [0, 1]$: 0 = "let nothing through", 1 = "let everything through".

---

## 3. Step-by-Step LSTM Walkthrough

### 3.1 Forget Gate
Decides what information to throw away from the cell state.
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
*   **Input:** Previous hidden state $h_{t-1}$ and current input $x_t$.
*   **Output:** number between 0 and 1 for each number in the cell state $C_{t-1}$.

### 3.2 Input Gate
Decides what new information to store in the cell state. This has two parts:
1.  **Sigmoid Layer:** Decides *which* values to update ($i_t$).
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
2.  **Tanh Layer:** Creates a vector of new candidate values ($\tilde{C}_t$).
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

### 3.3 Update Cell State
Update the old cell state $C_{t-1}$ into the new cell state $C_t$.
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$
*   Multiply old state by $f_t$ (forgetting things).
*   Add $i_t * \tilde{C}_t$ (adding new candidate values, scaled by how much we decided to update each state value).

### 3.4 Output Gate
Decides what to output. This output will be a filtered version of the cell state.
1.  **Sigmoid Layer:** Decides what parts of the cell state to output ($o_t$).
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
2.  **Filter:** Put cell state through $\tanh$ (to push values between -1 and 1) and multiply by output of sigmoid gate.
    $$ h_t = o_t * \tanh(C_t) $$

---

## 4. Variants on LSTM

### 4.1 Peephole Connections
Allow gate layers to look at the cell state.
$$ f_t = \sigma(W_f \cdot [C_{t-1}, h_{t-1}, x_t] + b_f) $$

### 4.2 Coupled Forget and Input Gates
Decide what to forget and what to add simultaneously. We only forget when we are going to input something in its place.
$$ C_t = f_t * C_{t-1} + (1 - f_t) * \tilde{C}_t $$

### 4.3 Gated Recurrent Unit (GRU)
Combines the forget and input gates into a single "update gate". It also merges the cell state and hidden state. The resulting model is simpler than standard LSTM models.
*   **Update Gate ($z_t$):** Controls how much past info to keep.
*   **Reset Gate ($r_t$):** Controls how much to forget.
