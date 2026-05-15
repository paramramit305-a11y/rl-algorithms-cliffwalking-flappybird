# Reinforcement Learning — From Scratch to Flappy Bird

A progressive implementation of core Reinforcement Learning algorithms, starting from tabular methods (Q-Learning, SARSA) on the classic CliffWalking environment, all the way to a Deep Q-Network (DQN) that learns to play Flappy Bird.

---

## Implementations

### 1. Q-Learning — CliffWalking
**File:** `Q_Learning.ipynb`

Off-policy TD algorithm. Agent learns the optimal (shortest) path from S to E while navigating near the cliff.

- Environment: `CliffWalking-v1` (Gymnasium)
- Algorithm: Q-Learning with ε-greedy exploration
- Result: **-13 reward, 13 steps** (optimal cliff-edge path)

### 2. SARSA — CliffWalking
**File:** `SARSA.ipynb`

On-policy TD algorithm. Agent learns a safer, longer path by accounting for exploration risk.

- Environment: `CliffWalking-v1` (Gymnasium)
- Algorithm: SARSA with ε-greedy exploration
- Result: **-17 reward, 17 steps** (safe path away from cliff)

**Key difference from Q-Learning:**
| | Q-Learning | SARSA |
|---|---|---|
| Policy | Off-policy | On-policy |
| Path | Cliff-edge shortcut | Safe longer path |
| Next action | `max Q(s', a)` | `Q(s', a')` — actual next action |

### 3. Deep Q-Network (DQN) — Flappy Bird
**Files:** `agent.py`, `dqn.py`, `experience_replay.py`, `parameters.yaml`, `game_flappy_bird.py`

Full DQN implementation with Experience Replay and Target Network, trained on Flappy Bird.

- Environment: `FlappyBird-v0` (flappy-bird-gymnasium)
- State: 12 LIDAR sensor values
- Actions: 0 = do nothing, 1 = flap
- Neural Network: `Linear(12→256, ReLU) → Linear(256→2)`
- Best reward achieved: **21.2** (episode 32,607)

---

## DQN Architecture

```
State (12) → Linear → ReLU → Linear → Q-values (2 actions)
               256 hidden nodes
```

Two key innovations over basic Q-Learning:

**Experience Replay** — Stores past experiences `(state, action, reward, next_state, terminated)` in a replay buffer. Randomly samples mini-batches during training to break correlation between sequential experiences.

**Target Network** — A frozen copy of the policy network used to compute stable TD targets. Synced with the policy network every 10 steps.

---

## Training Progress (Flappy Bird)

| Episode | Best Reward |
|---------|------------|
| 1       | -6.9       |
| 73      | -0.9       |
| 1031    | +0.3       |
| 4411    | +4.3       |
| 10348   | +11.1      |
| 25363   | +21.1      |
| 32607   | +21.2      |

---

## Project Structure

```
├── Q_Learning.ipynb          # Q-Learning on CliffWalking
├── SARSA.ipynb               # SARSA on CliffWalking
├── agent.py                  # DQN Agent (train + test)
├── dqn.py                    # Neural Network definition
├── experience_replay.py      # Replay Buffer
├── game_flappy_bird.py       # Play Flappy Bird manually
├── parameters.yaml           # Hyperparameters
└── runs/
    ├── flappybirdv0.pt       # Trained model weights
    └── flappybirdv0.log      # Training log
```

---

## Hyperparameters (DQN)

```yaml
alpha: 0.001
gamma: 0.99
epsilon_init: 1.0
epsilon_min: 0.05
epsilon_decay: 0.9995
replay_memory_size: 100000
mini_batch_size: 32
network_sync_rate: 10
reward_threshold: 1000
```

---

## Setup

```bash
pip install gymnasium
pip install gymnasium[toy-text]
pip install flappy-bird-gymnasium
pip install torch pygame
```

**Play Flappy Bird manually:**
```bash
python game_flappy_bird.py
```

**Test trained DQN agent:**
```bash
python agent.py flappybirdv0
```

**Train from scratch:**
```bash
python agent.py flappybirdv0 --train
```

---

## Concepts Covered

- Markov Decision Process (MDP)
- Temporal Difference (TD) Learning
- Q-Learning vs SARSA (off-policy vs on-policy)
- Deep Q-Networks (DQN)
- Experience Replay
- Target Network
- ε-greedy Exploration with Decay

---

## Tech Stack

`Python` `PyTorch` `Gymnasium` `NumPy` `Pygame`

---

## Author

**Parmar Amit**

[![GitHub](https://img.shields.io/badge/GitHub-paramramit305--a11y-181717?style=flat&logo=github)](https://github.com/paramramit305-a11y)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-parmar--amit-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/parmar-amit-97941a377)
