# NLP Fundamentals and Recurrent Neural Networks

## 1. Natural Language Processing (NLP) Overview
Natural Language Processing (NLP) is the intersection of computer science, artificial intelligence, and linguistics. It involves the interaction between computers and human language.

### Key Preprocessing Steps
*   **Tokenization:** Breaking text into individual words or subwords (tokens).
*   **Normalization:** Converting text to a standard format (e.g., lowercasing).
*   **Lemmatization:** Reducing words to their base or dictionary form (e.g., "running" $ \rightarrow $ "run").
*   **Stemming:** Cutting off prefixes/suffixes to find the root form (e.g., "running" $ \rightarrow $ "run", but also "better" $ \rightarrow $ "bet" - crude heuristic).

---

## 2. Word Embedding Techniques
Traditional representations like One-Hot Encoding result in sparse, high-dimensional vectors that capture no semantic meaning. Word Embeddings solve this by mapping words to dense, lower-dimensional vectors where similar words are close in space.

### 2.1 Frequency-Based (Bag of Words / TF-IDF)
*   Represent documents based on word counts.
*   **TF-IDF:** Weights words by how unique they are to a document relative to the corpus.

### 2.2 Prediction-Based (Word2Vec)
Learns embeddings by predicting context.
*   **CBOW (Continuous Bag of Words):** Predict target word from surrounding context words.
*   **Skip-Gram:** Predict surrounding context words from a target word.

### 2.3 GloVe (Global Vectors)
Matrix factorization method that learns from the global co-occurrence statistics of words in the corpus.

---

## 3. Recurrent Neural Networks (RNN)
RNNs are designed for sequential data. Unlike Feed-Forward Networks, they have a "memory" (hidden state) which captures information about what has been calculated so far.

### 3.1 Architecture
The hidden state $h_t$ at time step $t$ is calculated as:
$$ h_t = \tanh(W_h h_{t-1} + W_x x_t + b) $$
The output $y_t$:
$$ y_t = W_y h_t + b_y $$

### 3.2 The Vanishing Gradient Problem
During backpropagation through time (BPTT), gradients can become extremely small, preventing weights from updating effectively. This limits RNNs' ability to learn long-range dependencies.

---

## 4. Long Short-Term Memory (LSTM)
LSTMs were designed to solve the vanishing gradient problem. They introduce a **Cell State ($C_t$)** (long-term memory) and three gates to regulate information flow.

### 4.1 Gates
1.  **Forget Gate ($f_t$):** Decides what to throw away from the cell state.
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
2.  **Input Gate ($i_t$):** Decides what new information to store.
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$
3.  **Output Gate ($o_t$):** Decides what to output (hidden state).
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
    $$ h_t = o_t * \tanh(C_t) $$

### 4.2 Cell State Update
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$

---

## 5. Gated Recurrent Unit (GRU)
A simplified version of LSTM that merges the cell state and hidden state, and typically trains faster.

### 5.1 Gates
1.  **Update Gate ($z_t$):** Determines how much of the past information needs to be passed along to the future (combines input and forget gates).
    $$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) $$
2.  **Reset Gate ($r_t$):** Determines how much of the past information to forget.
    $$ r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) $$

### 5.2 Hidden State Update
$$ \tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t]) $$
$$ h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t $$

---

## Comparisons
| Feature | RNN | LSTM | GRU |
| :--- | :--- | :--- | :--- |
| **Memory** | Short-term | Long & Short-term | Long & Short-term |
| **Complexity** | Low | High (3 gates, cell state) | Medium (2 gates) |
| **Speed** | Fast | Slower | Faster than LSTM |
| **Use Case** | Short sequences | Complex, long dependencies | Efficient long dependencies |
