# Sequence-to-Sequence (Seq2Seq) and Attention Mechanism

## 1. Introduction
Sequence-to-Sequence (Seq2Seq) models are neural architectures designed to map one sequence to another sequence, possibly of different lengths. These models are widely used in:
*   Machine Translation
*   Text Summarization
*   Paraphrasing
*   Generative Question Answering

A Seq2Seq model consists of two recurrent neural networks (typically LSTMs):
*   **Encoder LSTM:** Reads and understands the input sequence.
*   **Decoder LSTM:** Generates the output sequence based on the encoded meaning.

---

## 2. Problem Setup and Running Example

Consider the input sentence:
> **Input:** "I love machine learning course"

After tokenization:
$$ [I, love, machine, learning, course] $$

Assume a 2-dimensional word embedding (for illustration only):
*   I $\rightarrow [0.1, 0.8]$
*   love $\rightarrow [0.9, 0.4]$
*   machine $\rightarrow [0.6, 0.7]$
*   learning $\rightarrow [0.7, 0.6]$
*   course $\rightarrow [0.3, 0.2]$

Thus, the input to the encoder LSTM is:
$$ X = [x_1, x_2, x_3, x_4, x_5], \quad x_i \in \mathbb{R}^2 $$

---

## 3. Encoder LSTM: Encoding the Input Sequence

### 3.1 Role of the Encoder
The encoder LSTM reads the input sequence one word at a time and converts it into a sequence of hidden states. Its only goal is to **understand** the input sentence. It does *not* generate any output words.

### 3.2 Step-by-Step Encoder Computation
At each time step $t$, the encoder computes:
$$ h_t, c_t = \text{LSTM}_{enc}(x_t, h_{t-1}, c_{t-1}) $$

**Table 1: Encoder Hidden State Computation**

| Time Step | Input Word | Embedding | Hidden State |
| :--- | :--- | :--- | :--- |
| $t=1$ | I | $x_1$ | $h_1$ |
| $t=2$ | love | $x_2$ | $h_2$ |
| $t=3$ | machine | $x_3$ | $h_3$ |
| $t=4$ | learning | $x_4$ | $h_4$ |
| $t=5$ | course | $x_5$ | $h_5$ |

Each hidden state $h_t$ summarizes all words seen up to time $t$.

### 3.3 Context Vector and Bottleneck
After processing the full sentence, the final encoder hidden state $h_5$ is treated as the **context vector**:
$$ \text{Context Vector} = h_5 $$
This vector represents the entire meaning of the input sentence and is passed to the decoder.

---

## 4. Decoder LSTM: Generating the Output Sequence

### 4.1 Role of the Decoder
The decoder LSTM converts the encoded meaning into an output sentence. For example:
> **Output:** "I enjoy ML class"

The decoder never sees the input words directly.

### 4.2 Decoder Initialization
The decoder is initialized using the final encoder states:
$$ s_0 = h_5 $$
$$ c_0 = c_5 $$

### 4.3 Decoder Inputs
At each decoding step:
*   At $t=1$: input token is `<BOS>` (Beginning of Sentence).
*   At $t>1$: input token is the previously generated word ($y_{t-1}$).

### 4.4 Decoder Computation Without Attention
At time step $t$:
$$ s_t, c_t = \text{LSTM}_{dec}(y_{t-1}, s_{t-1}, c_{t-1}) $$

Word prediction is performed outside the LSTM:
$$ \hat{y}_t = \text{Softmax}(W s_t + b) $$

**Table 2: Decoder Generation Process**

| Time | Decoder Input | Hidden State | Output Word |
| :--- | :--- | :--- | :--- |
| $t=1$ | `<BOS>` | $s_1$ | I |
| $t=2$ | I | $s_2$ | enjoy |
| $t=3$ | enjoy | $s_3$ | ML |
| $t=4$ | ML | $s_4$ | class |
| $t=5$ | class | $s_5$ | `<EOS>` |

---

## 5. Limitations of Simple Seq2Seq

### 5.1 Information Bottleneck
All information must pass through a **single vector** $h_5$. For long or complex sentences, this causes information loss.

### 5.2 No Explicit Alignment
The model does not know which input words influence which output words. For example:
*   `machine learning` $\rightarrow$ `ML`
There is no explicit alignment between encoder and decoder words.

### 5.3 Poor Long-Sequence Performance
As sentence length increases, early information is forgotten, degrading output quality.

---

## 6. Attention Mechanism

### 6.1 Motivation
Instead of relying on a single context vector, attention allows the decoder to **dynamically access all encoder hidden states** at each decoding step.

### 6.2 Encoder Memory
The encoder produces a set of memory vectors:
$$ \{h_1, h_2, h_3, h_4, h_5\} $$

### 6.3 Decoder State as Query
At decoder step $t=1$:
$$ s_1, c_1 = \text{LSTM}_{dec}(\text{<BOS>}, h_5, c_5) $$
Here, $s_1$ is not a prediction; it acts as the decoder’s **internal query**.

### 6.4 Attention Score Computation
We calculate how relevant each encoder state $h_i$ is to the current decoder state $s_1$:
$$ \text{score}_{1,i} = s_1^\top h_i $$

### 6.5 Softmax Normalization
We normalize scores to get probabilities (attention weights):
$$ \alpha_{1,i} = \frac{\exp(\text{score}_{1,i})}{\sum_{j=1}^5 \exp(\text{score}_{1,j})} $$

### 6.6 Context Vector
The weighted sum of encoder states forms the context vector for this step:
$$ c^{att}_1 = \sum_{i=1}^5 \alpha_{1,i} h_i $$

### 6.7 Final Prediction
We concatenate the decoder state with the context vector to predict the word:
$$ \tilde{s}_1 = [s_1; c^{att}_1] $$
$$ \hat{y}_1 = \text{Softmax}(W \tilde{s}_1 + b) $$

---

## 7. Summary
*   **Simple Seq2Seq** suffers from bottleneck and alignment issues.
*   **Attention** enables dynamic encoder focus.
*   **Decoder states** act as queries, not just predictions.
*   **Prediction** happens *after* attention is computed.
