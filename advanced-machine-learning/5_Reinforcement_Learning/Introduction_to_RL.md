# Introduction to Reinforcement Learning (RL)

Reinforcement Learning is a subfield of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** and receiving feedback in the form of **rewards**. Unlike supervised learning, where the correct answer is provided, the agent must discover the best strategy through trial and error to maximize the cumulative reward over time.

---

## 1. RL Algorithms
RL algorithms can be categorized based on how they process information and what they optimize:

### By Model Knowledge
*   **Model-Based:** The agent builds a model of the environment (transition probabilities and reward functions) and uses it to plan (e.g., Dynamic Programming).
*   **Model-Free:** The agent learns directly from experience (samples of state-action-reward) without knowing the underlying dynamics (e.g., Q-Learning, Policy Gradients).

### By Optimization Target
*   **Value-Based:** The agent learns a value function $V(s)$ or $Q(s, a)$ to estimate the expected return, then picks actions to maximize this value (e.g., DQN, SARSA).
*   **Policy-Based:** The agent learns the policy $\pi(a|s)$ directly, optimizing the mapping from state to action (e.g., REINFORCE).
*   **Actor-Critic:** Combines both methods—an "Actor" (policy) decides actions, and a "Critic" (value function) evaluates them.

---

## 2. Elements of Reinforcement Learning

The core components of an RL problem are:

1.  **Agent:** The learner and decision-maker.
2.  **Environment:** The external system the agent interacts with.
3.  **Policy ($\pi$):** The agent's behavior strategy. It maps states to actions (or probabilities of actions).
    *   *Deterministic:* $a = \pi(s)$
    *   *Stochastic:* $\pi(a|s) = P(A_t=a | S_t=s)$
4.  **Reward Signal ($R_t$):** A scalar number sent from the environment to the agent at each time step. It defines the goal: maximize the total cumulative reward.
5.  **Value Function:** A prediction of future rewards. It specifies what is "good" in the long run.
    *   *State-Value $V(s)$:* How good it is to be in state $s$.
    *   *Action-Value $Q(s, a)$:* How good it is to take action $a$ in state $s$.
6.  **Model (Optional):** A representation of how the environment works (e.g., physics engine, rules of the game).

---

## 3. Markov Decision Process (MDP) and Dynamic Programming

To formalize RL problems mathematically, we use the framework of Markov Decision Processes.

### 3.1 Markov Chain and Markov Process

**Markov Property**
A state $S_t$ is **Markov** if and only if:
$$ P[S_{t+1} | S_t] = P[S_{t+1} | S_1, ..., S_t] $$
*“The future is independent of the past given the present.”*

**Markov Process (MP)**
A Markov Process is a tuple $<S, P>$:
*   $S$: A set of states.
*   $P$: The state transition probability matrix, where $P_{ss'} = P[S_{t+1}=s' | S_t=s]$.
*   It represents a memoryless random sequence of states.

**Markov Reward Process (MRP)**
An MRP adds value to the chain. It is a tuple $<S, P, R, \gamma>$:
*   $R$: Reward function, $R_s = E[R_{t+1} | S_t = s]$.
*   $\gamma$: Discount factor $\in [0, 1]$.

---

### 3.2 Markov Decision Process (MDP)

An MDP adds **Agency** (decisions/actions) to the MRP. It is the standard formalism for RL.

**Definition:** An MDP is a tuple $<S, A, P, R, \gamma>$:
*   $S$: A finite set of states.
*   $A$: A finite set of actions.
*   $P$: State transition probability matrix:
    $$ P_{ss'}^a = P[S_{t+1} = s' | S_t = s, A_t = a] $$
*   $R$: Reward function:
    $$ R_s^a = E[R_{t+1} | S_t = s, A_t = a] $$
*   $\gamma$: Discount factor ($\gamma \in [0, 1]$). It determines the importance of future rewards.

**Goal:**
The goal in an MDP is to find an optimal policy $\pi^*$ that maximizes the **Return** ($G_t$), which is the total discounted reward from time step $t$:
$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$

---

### 3.3 Dynamic Programming (DP)

Dynamic Programming corresponds to a collection of algorithms used to compute optimal policies given a **perfect model** of the environment (i.e., we know $S, A, P, R, \gamma$).

**Bellman Expectation Equation (The Recursive Definition)**
The value of a state is the immediate reward plus the discounted value of the next state:
$$ V_\pi(s) = \sum_{a \in A} \pi(a|s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_\pi(s') \right) $$

**Standard DP Algorithms:**
1.  **Policy Evaluation:** Iteratively compute $V_\pi$ for a specific policy $\pi$.
2.  **Policy Iteration:** Alternate between evaluating the current policy and improving it (greedily).
3.  **Value Iteration:** Directly compute the optimal value function $V^*$ by applying the Bellman Optimality Equation iteratively.
