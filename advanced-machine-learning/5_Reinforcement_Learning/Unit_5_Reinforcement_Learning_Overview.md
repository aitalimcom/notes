# Unit 5: Reinforcement Learning

## 5.1 Fundamentals of Reinforcement Learning

Reinforcement Learning (RL) is a subfield of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** and receiving feedback in the form of **rewards**. Unlike supervised learning, where the correct answer is provided, the agent must discover the best strategy through trial and error to maximize the cumulative reward over time.

### RL Algorithms Categorization
*   **Model-Based:** The agent builds a model of the environment (transition probabilities and reward functions) and uses it to plan (e.g., Dynamic Programming).
*   **Model-Free:** The agent learns directly from experience (samples of state-action-reward) without knowing the underlying dynamics (e.g., Q-Learning, Policy Gradients).
*   **Value-Based:** The agent learns a value function $V(s)$ or $Q(s, a)$ to estimate expected return (e.g., DQN, SARSA).
*   **Policy-Based:** The agent learns the policy $\pi(a|s)$ directly (e.g., REINFORCE).
*   **Actor-Critic:** Combines both methods.

### Elements of RL
1.  **Agent:** The learner and decision-maker.
2.  **Environment:** The external system.
3.  **Policy ($\pi$):** The agent's behavior strategy.
4.  **Reward Signal ($R_t$):** A scalar feedback used to evaluate the agent's action.
5.  **Value Function:** A prediction of future rewards.
6.  **Model (Optional):** A representation of how the environment works.

---

## 5.2 Markov Model and Decision Processes

### Markov Property
A state $S_t$ is **Markov** if and only if:
$$ P[S_{t+1} | S_t] = P[S_{t+1} | S_1, ..., S_t] $$
*“The future is independent of the past given the present.”*

### Markov Decision Process (MDP)
An MDP is a tuple $<S, A, P, R, \gamma>$:
*   $S$: Finite set of states.
*   $A$: Finite set of actions.
*   $P$: State transition probability matrix ($P_{ss'}^a$).
*   $R$: Reward function ($R_s^a$).
*   $\gamma$: Discount factor ($\gamma \in [0, 1]$).

The goal is to find an optimal policy $\pi^*$ that maximizes the expected return $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.

---

## 5.3 Dynamic Programming

Dynamic Programming (DP) algorithms compute optimal policies given a **perfect model** of the environment.

### Bellman Equation
The Bellman equation expresses a recursive relationship between the value of a state and the values of its successor states:
$$ V_\pi(s) = \sum_{a \in A} \pi(a|s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_\pi(s') \right) $$

### Value Iteration
**Value iteration** starts with an arbitrary value function and repeatedly improves it until convergence.
1.  Initialize $V(s)$ arbitrarily.
2.  Repeat until convergence:
    $$ V(s) \leftarrow \max_a \left( R_s^a + \gamma \sum_{s'} P_{ss'}^a V(s') \right) $$
3.  Output a deterministic policy $\pi(s) = \arg\max_a Q(s,a)$.

### Policy Iteration
**Policy iteration** alternates between:
1.  **Policy Evaluation:** Compute $V_\pi(s)$ for current $\pi$.
2.  **Policy Improvement:** Update $\pi(s) = \arg\max_a \left( R_s^a + \gamma \sum_{s'} P_{ss'}^a V_\pi(s') \right)$.

---

## 5.4 Model Free Algorithms

When the environment's dynamics ($P$ and $R$) are unknown, agents must learn from experience (samples).

### Temporal Difference (TD) Learning
TD learning combines Monte Carlo ideas (learning from experience) and DP ideas (bootstrapping). The agent updates estimates based in part on other learned estimates, without waiting for a final outcome.

### SARSA (State-Action-Reward-State-Action)
SARSA is an **On-Policy** algorithm. It learns the value of the policy being carried out by the agent (including the exploration steps).
*   **Update Rule:**
    $$ Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)] $$
*   Here, both $A$ (current action) and $A'$ (next action) are chosen by the current policy (e.g., $\epsilon$-greedy).

### Q-Learning
Q-Learning is an **Off-Policy** algorithm. It learns the value of the *optimal* policy, independently of the agent's actions.
*   **Update Rule:**
    $$ Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a'} Q(S', a') - Q(S, A)] $$
*   The target uses the maximum Q-value for the next state, assuming the best action is taken, regardless of what action the agent actually takes.

---

## 5.5 Policy Gradient Methods

Value-based methods (like Q-Learning) infer a policy from a value function. **Policy Gradient** methods learn the policy parameters $\theta$ directly to maximize the objective function $J(\theta)$ (expected return).

### REINFORCE Algorithm (Monte Carlo Policy Gradient)
REINFORCE relies on an estimated return by Monte Carlo methods (using full episode returns) to update the policy parameters.
1.  Initialize policy parameters $\theta$.
2.  Generate an episode $S_0, A_0, R_1, ..., S_T$ following $\pi_\theta$.
3.  For each step $t$ in the episode:
    *   Calculate return $G_t$ from step $t$.
    *   Update $\theta$:
        $$ \theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \ln \pi_\theta(A_t | S_t) $$

---

## 5.6 Deep Reinforcement Learning

Deep RL approximates value functions or policies using Deep Neural Networks, allowing RL to scale to complex, high-dimensional state spaces.

### Deep Q-Network (DQN)
DQN adapts Q-Learning by using a neural network $Q(s, a; \theta)$ to approximate the Q-value function.

**Key Innovations:**
1.  **Experience Replay:** Stores transitions $(s, a, r, s')$ in a buffer and samples minibatches randomly for training. This breaks correlations between consecutive samples and stabilizes training.
2.  **Target Network:** Uses a separate network ($\hat{Q}$) with fixed parameters to calculate the target values. The weights of $\hat{Q}$ are updated periodically from the main network.
3.  **Reward Clipping:** Normalizes rewards (e.g., -1, 0, 1) to ensure gradients remain well-conditioned.

**Algorithm:**
1.  Input state (e.g., image pixels).
2.  Q-Network outputs Q-values for all actions.
3.  Select action using $\epsilon$-greedy.
4.  Store experience in Memory.
5.  Sample minibatch and minimize Loss:
    $$ L(\theta) = E [(R + \gamma \max_{a'} \hat{Q}(s', a') - Q(s, a; \theta))^2] $$

---

## 5.7 Practical Applications

1.  **Game Playing:** AlphaGo (Go), OpenAI Five (Dota 2), StarCraft II agents.
2.  **Robotics:** Robot manipulation, walking, and navigation tasks where modeling complex physics is difficult.
3.  **Autonomous Driving:** Decision making for lane changes, overtaking, and intersection handling.
4.  **Finance:** Algorithmic trading and portfolio management.
5.  **Recommendation Systems:** Optimizing long-term user engagement rather than immediate clicks.
