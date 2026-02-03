# Value Functions, Dynamic Programming, and Deep Q Networks

## 1. State-Action Value Function (Q Function)

In Reinforcement Learning (RL), the **state-action value function**, commonly called the **Q function**, measures how good it is for an agent to take a particular action in a given state while following a policy.

Unlike the value function, which evaluates only states, the Q function evaluates state–action pairs. This makes it more informative, as it directly supports action selection.

### 1.1 Conceptual Meaning
The Q function represents the expected future reward obtained when:
*   The agent starts in a state,
*   Takes a specific action,
*   Continues following a given policy.

Thus, it answers the question:
> "How good is it to take this action in this state under a policy?"

### 1.2 Difference Between Value Function and Q Function
*   **Value Function:** Measures the goodness of a *state*.
*   **Q Function:** Measures the goodness of an *action in a state*.

Because of this, Q functions are widely used in action-selection and control problems.

### 1.3 Q Table Representation
The Q function can be represented as a table, known as a **Q table**, where each entry corresponds to a state–action pair. The agent selects the action with the highest value in each state.

In practice, whenever we refer to a value function or a Q function, we often mean their tabular representations.

---

## 2. Bellman Equation and Optimality

The **Bellman equation**, introduced by Richard Bellman, is central to Reinforcement Learning. It expresses a recursive relationship between the value of a state (or state–action pair) and the values of its successor states.

Solving a Markov Decision Process (MDP) means finding:
*   An optimal value function, and
*   An optimal policy.

### 2.1 Optimal Value Function and Policy
Different policies produce different value functions. The **optimal value function** is the one that yields the maximum expected return across all policies.

An **optimal policy** is the policy that results in this optimal value function.

### 2.2 Bellman Optimality Principle
The optimal value of a state is obtained by choosing the action that has the highest Q value in that state. This idea leads to the **Bellman optimality equation**, which forms the theoretical foundation of many reinforcement learning algorithms.

---

## 3. Value Iteration

**Value iteration** is a dynamic programming method used to compute the optimal value function. It starts with an arbitrary value function and repeatedly improves it until convergence.

### 3.1 Key Idea
The initial value function may not be optimal. Therefore, the algorithm repeatedly updates it by selecting better estimates based on expected future rewards until the changes become negligible.

### 3.2 Steps in Value Iteration
1.  **Initialize** a random value for each state.
2.  **Compute** the Q function for all state–action pairs.
3.  **Update** the value function by selecting the maximum Q value for each state.
4.  **Repeat** the process until the value function stabilizes.

### 3.3 Intuitive Understanding
Through repeated updates, the value function gradually improves. Each iteration uses the updated values from the previous iteration, allowing information about future rewards to propagate backward through the state space.

Once the value function converges, the optimal policy can be derived by selecting, in each state, the action that yields the highest Q value.

---

## 4. Policy Iteration

**Policy iteration** is another dynamic programming approach, but unlike value iteration, it starts with a random policy rather than a value function.

### 4.1 Key Idea
The algorithm alternates between evaluating a policy and improving it. This process continues until the policy no longer changes.

### 4.2 Two Main Steps
*   **Policy Evaluation:** Compute the value function for the current policy.
*   **Policy Improvement:** Update the policy by selecting better actions based on the computed value function.

### 4.3 Steps in Policy Iteration
1.  **Initialize** a random policy.
2.  **Evaluate** the value function of the policy.
3.  **Improve** the policy using the updated value function.
4.  **Repeat** until the policy converges.

Policy iteration often converges in **fewer iterations** than value iteration but requires solving the value function more accurately at each step.

---

## 5. Deep Q Network (DQN)

Traditional Q learning uses a Q table, which works well only when the number of states and actions is small. In complex environments with large or continuous state spaces, maintaining a Q table becomes impractical.

**Deep Q Networks (DQN)** address this problem by approximating the Q function using neural networks.

### 5.1 Motivation
Instead of storing Q values explicitly, DQN uses a neural network to estimate Q values for all possible actions given a state. This allows reinforcement learning to scale to high-dimensional environments such as video games.

### 5.2 Basic Idea of DQN
*   The neural network takes the **state** as input.
*   It outputs **Q values** for all possible actions.
*   The action with the highest Q value is selected.

The network is trained by minimizing the difference between predicted and target Q values using gradient descent.

---

## 6. Architecture of Deep Q Network

### 6.1 Convolutional Neural Network
For environments like Atari games, the input to the DQN is a game screen image. Since raw images are large and computationally expensive, they are preprocessed by:
*   Downsampling the image,
*   Converting it to grayscale.

The processed images are fed into **convolutional layers** that capture spatial relationships between objects in the game.

**Pooling layers are not used** because the precise position of objects is important for decision-making in games.

### 6.2 Multiple Frames as Input
To capture motion information, DQN uses **multiple consecutive frames** as input. This allows the network to infer movement direction and velocity, which cannot be determined from a single frame.

### 6.3 Output Layer
The output layer contains **one unit per action**. This design allows the network to compute Q values for all actions in a single forward pass.

---

## 7. Experience Replay

During interaction with the environment, the agent stores experiences in the form of `(state, action, reward, next_state)` in a **replay buffer**.

### 7.1 Why Experience Replay?
*   **Reduces correlation** between consecutive experiences.
*   **Prevents overfitting** caused by sequential data.
*   **Improves learning stability** by sampling diverse experiences.

The replay buffer stores a fixed number of recent experiences and discards older ones as new data arrives.

---

## 8. Target Network

Using the same network to compute both predicted and target Q values can lead to **instability** during training.

To address this, DQN introduces a separate **target network**:
*   The target network is used **only** to compute target Q values.
*   Its parameters are **periodically updated** from the main network.

This separation stabilizes learning and reduces divergence.

---

## 9. Reward Clipping

Reward scales vary across environments. To maintain stable learning, rewards are clipped to a fixed range, typically between **-1 and +1**. This prevents large reward values from dominating the learning process.

---

## 10. Overall DQN Algorithm

The working of DQN can be summarized as follows:

1.  **Observe** the current state and preprocess it.
2.  **Select** an action using an epsilon-greedy strategy.
3.  **Execute** the action and observe the reward and next state.
4.  **Store** the experience in the replay buffer.
5.  **Sample** random experiences and compute the training loss.
6.  **Update** network parameters using gradient descent.
7.  **Periodically update** the target network.
8.  **Repeat** for multiple episodes.

---

## 11. Conclusion

**Value iteration** and **policy iteration** provide foundational dynamic programming methods for solving reinforcement learning problems. **Deep Q Networks** extend these ideas to complex environments by combining Q learning with deep neural networks. Together, these methods form the backbone of modern value-based reinforcement learning.
