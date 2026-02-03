# NLP: Seq2Seq, Attention, and Transformers

*Source: Lena Voita's NLP Course*

## 1. Sequence-to-Sequence (Seq2Seq) Basics

The goal of machine translation (and general seq2seq tasks) is to find the target sequence $Y$ that maximizes the conditional probability given the input sequence $X$.
$$ \hat{Y} = \arg \max_Y P(Y|X; \theta) $$

### 1.1 Encoder-Decoder Framework
*   **Encoder:** Reads source sequence and produces a representation (context).
*   **Decoder:** Uses source representation to generate target sequence.

### 1.2 Conditional Language Models
Seq2Seq models are **Conditional Language Models**.
*   Standard LM: $P(y_t | y_{<t})$
*   Seq2Seq: $P(y_t | y_{<t}, X)$ (Conditioned on source $X$)

### 1.3 The Simplest Model: RNNs
*   **Encoder RNN:** Reads source sentence token by token. The final hidden state "encodes" the whole sentence.
*   **Decoder RNN:** Initialized with the encoder's final state. Generates target tokens.
*   **Training:** Cross-Entropy Loss to maximize probability of correct next token. $\mathcal{L} = - \sum \log P(y_t^{true} | y_{<t}, x)$.

### 1.4 Inference
We cannot check all possible sequences (infinite). We use approximations:
*   **Greedy Decoding:** Pick most probable token at each step. Fast but suboptimal.
*   **Beam Search:** Keep track of top-$K$ hypotheses (beam size) at each step. Better quality.

---

## 2. Attention Mechanism

**Problem with Fixed Encoder:** Compressing a long sentence into a *single fixed vector* is hard (information bottleneck).

**Solution:** Allow the decoder to "look at" different parts of the source at each step.

### 2.1 High-Level View
At each decoder step $t$:
1.  **Input:** Decoder state $h_t$ and all encoder states $s_1, ..., s_m$.
2.  **Scores:** Compute relevance $score(h_t, s_i)$ for each source token.
3.  **Weights:** Apply Softmax to scores to get probabilities $\alpha_{ti}$.
4.  **Context:** Compute weighted sum of encoder states $c_t = \sum \alpha_{ti} s_i$.
5.  **Predict:** Use $c_t$ to predict next token.

### 2.2 Score Functions
*   **Dot-Product:** $h_t^T s_i$
*   **Bilinear (Luong):** $h_t^T W s_i$
*   **MLP (Bahdanau):** $v^T \tanh(W_1 h_t + W_2 s_i)$

### 2.3 Key Variants
*   **Bahdanau Attention:**
    *   Encoder: Bidirectional RNN.
    *   Score: MLP.
    *   Applied: *Between* steps (before decoder update).
*   **Luong Attention:**
    *   Score: Bilinear / Dot.
    *   Applied: *After* decoder RNN step (before prediction).

---

## 3. Transformer: "Attention is All You Need"

Dropped recurrence (RNNs) entirely. Relies solely on **Self-Attention**.
*   **Benefits:** Parallelizable (faster training), captures long-range dependencies better ($O(1)$ path length).

### 3.1 Self-Attention
Tokens interact with each other to gather context.
*   **Query (Q):** Asking for info.
*   **Key (K):** Holding info used for matching.
*   **Value (V):** The actual info content.

$$ Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

### 3.2 Architectures Details
*   **Masked Self-Attention:** In Decoder, prevents looking at future tokens during training.
*   **Multi-Head Attention:** Multiple independent attention heads to focus on different relations (e.g., syntactic, semantic, positional).
*   **Feed-Forward Networks:** MLP applied to each token independently after attention.
*   **Add & Norm:** Residual connections + Layer Normalization.
*   **Positional Encoding:** Since there's no RNN, we add vectors to embeddings to give order information (Sine/Cosine functions).

---

## 4. Subword Segmentation: Byte Pair Encoding (BPE)

**Problem:** Fixed vocabulary leads to many "UNK" (unknown) tokens.
**Solution:** Break rare words into subwords (e.g., "unrelated" -> "un", "relat", "ed").

### algorithm
1.  **Train:**
    *   Start with characters.
    *   Iteratively merge most frequent adjacent pair.
    *   Add new token to vocab.
2.  **Inference:** Apply learned merges to new text greedy.

---

## 5. Interpretability

**What do heads learn?**
*   **Positional:** Focus on neighbors.
*   **Syntactic:** Track subject-verb, verb-object dependencies.
*   **Rare Tokens:** Focus on rare words (sometimes overfitting).

**Pruning:** Many heads are redundant and can be removed after training without large quality loss.

**Probing:** Training classifiers on frozen representations shows:
*   Encoder layers capture POS tags.
*   Training on morphologically simpler target languages forces the encoder to learn *more* about source morphology.
