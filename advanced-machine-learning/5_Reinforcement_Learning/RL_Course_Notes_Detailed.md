# Reinforcement Learning: Detailed Course Notes & Implementation Guide

*Based on "Hands-On Reinforcement Learning with Python" and Course Syllabus*

## 1. Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a goal-oriented learning paradigm where an **agent** learns to make decisions by interacting with an **environment**.

### Key Analogy
*   **Dog Training**: You don't tell a dog exactly *how* to catch a ball (physics, muscle movement). You throw the ball; if the dog catches it, you give a **reward** (cookie). If it ignores it, no reward. The dog learns to repeat actions that yield cookies.
*   **RL Context**: The agent (dog) performs actions in the environment. Positive rewards reinforce behavior; negative rewards (punishments) discourage it.

### Core Elements
1.  **Agent**: The learner/decision-maker.
2.  **Environment**: Everything outside the agent.
3.  **State ($S_t$)**: Current situation of the agent.
4.  **Action ($A_t$)**: Move taken by the agent.
5.  **Reward ($R_t$)**: Numerical feedback signal.
6.  **Policy ($\pi$)**: Strategy mapping states to actions.
7.  **Value Function ($V(s)$)**: Expected long-term return from state $s$.
8.  **Model**: Internal representation of the environment (optional).

### Types of Environments
*   **Deterministic vs. Stochastic**: Is the next state certain or random?
*   **Fully vs. Partially Observable**: Can the agent see the whole board (Chess) or just a part (Poker)?
*   **Discrete vs. Continuous**: Finite moves (Chess) vs. infinite range (Robot arm angles).
*   **Episodic vs. Continuous**: Does the task end (Game over) or go on forever?

---

## 2. Markov Decision Process (MDP)

MDP provides the mathematical framework for RL.
*   **Markov Property**: "The future depends only on the present, not the past."
    $$ P[S_{t+1} | S_t] = P[S_{t+1} | S_1, ..., S_t] $$

### The MDP Tuple $<S, A, P, R, \gamma>$
*   $S$: States
*   $A$: Actions
*   $P$: Transition Probability
*   $R$: Reward Function
*   $\gamma$: **Discount Factor** (0 to 1). Determines importance of future rewards.
    *   $\gamma \approx 0$: Short-sighted (immediate rewards).
    *   $\gamma \approx 1$: Far-sighted (long-term returns).

### Bellman Equations
The foundation of solving MDPs.
*   **Value Function $V(s)$**:
    $$ V(s) = \max_a \sum_{s'} P_{ss'}^a [R_{ss'}^a + \gamma V(s')] $$
*   **Q-Function $Q(s, a)$**: value of taking action $a$ in state $s$.

---

## 3. Dynamic Programming (DP) & Solving FrozenLake

When the model ($P$ and $R$) is effectively known, we use DP.

### 3.1 Value Iteration
Iteratively update $V(s)$ until it converges to $V^*(s)$.

**Algorithm Concept:**
1.  Init $V(s) = 0$.
2.  Loop until convergence:
    *   For each state $s$:
        *   $V(s) \leftarrow \max_a \sum P(s'|s,a) [R + \gamma V(s')]$

**Python Implementation (FrozenLake):**
```python
import gym
import numpy as np

def value_iteration(env, gamma=1.0):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-20
    
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_values = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for prob, next_state, reward, _ in env.P[state][action]:
                    next_states_rewards.append(prob * (reward + gamma * updated_value_table[next_state]))
                Q_values.append(np.sum(next_states_rewards))
            value_table[state] = max(Q_values)
            
        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            break
            
    return value_table

def extract_policy(value_table, env, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward, _ in env.P[state][action]:
                Q_table[action] += prob * (reward + gamma * value_table[next_state])
        policy[state] = np.argmax(Q_table)
    return policy
```

### 3.2 Policy Iteration
Start with a random policy, evaluate it, improve it, repeat.
1.  **Evaluation**: Calculate $V_\pi$ for current policy.
2.  **Improvement**: Update $\pi$ to be greedy with respect to $V_\pi$.

---

## 4. Monte Carlo Methods

Used when the model is **unknown** (Model-Free). Learn from **episodes** of experience.
*   **Concept**: Approximate the mean return instead of expected return.
*   **First Visit MC**: Average returns only for the first time a state is visited in an episode.
*   **Every Visit MC**: Average returns for every visit.

### Example: Blackjack
Estimation of state values without knowing the dealer's deck probabilities, just by playing many hands.

```python
def first_visit_mc_prediction(policy, env, n_episodes):
    value_table = defaultdict(float)
    N = defaultdict(int) # Visit count
    
    for _ in range(n_episodes):
        states, _, rewards = generate_episode(policy, env)
        returns = 0
        # Iterate backwards
        for t in range(len(states)-1, -1, -1):
            R = rewards[t]
            S = states[t]
            returns += R
            # First visit check
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]
    return value_table
```

---

## 5. Temporal Difference (TD) Learning

Combines DP (bootstrapping) and MC (sampling). Updates estimates based on other estimates without waiting for the episode to end.

### 5.1 Q-Learning (Off-Policy)
Learns the value of the **optimal** policy, regardless of the agent's current actions (exploration).
$$ Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s,a)] $$

### 5.2 SARSA (On-Policy)
Learns the value of the **current** policy (including exploration noise).
$$ Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma Q(s', a') - Q(s,a)] $$

---

## 6. Deep Q Networks (DQN)

When state spaces are too large for tables (e.g., Atarai games with pixel inputs), use a Neural Network to approximate the Q-function.

### Key Components
1.  **Q-Network**: Input state -> Output Q-values for all actions.
2.  **Experience Replay**: Store transitions $(s, a, r, s')$ in a buffer. Sample random batches to train.
    *   *Why?* Breaks correlation between sequential steps, stabilizes training.
3.  **Target Network**: A copy of the Q-network used to calculate target values. Updated periodically.
    *   *Why?* Prevents "chasing a moving target" instability.
4.  **Reward Clipping**: Normalize rewards to $[-1, 1]$ to handle different game scales.

### Architecture (for Atari)
*   **Input**: Stack of 4 frames (grayscale, resized) to capture motion.
*   **Conv Layers**: Capture spatial features.
*   **FC Layers**: Output Q-values.
