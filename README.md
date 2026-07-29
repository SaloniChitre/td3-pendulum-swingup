# TD3 for Inverted Pendulum Swing-Up

Solution to Jefferson Lab's CST Data Scientist I Interview Problem #1 — solving the classic inverted pendulum swing-up problem using the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm on Gymnasium's `Pendulum-v1` environment.

## Problem

The pendulum starts in a random position on a fixed pivot and must be swung up and balanced upright — an unstable equilibrium — using continuous torque control applied directly at the pivot.

- **State space:** `[cos(θ), sin(θ), angular velocity]` — angle is represented as cosine/sine rather than a raw value to avoid the discontinuity at the 0°/360° wraparound point
- **Action space:** continuous torque, bounded to `[-2, 2]` N·m
- **Reward:** always negative (cost-based) — goal is to minimize the magnitude of the cost, not maximize a positive score

## Why TD3

The continuous action space rules out discrete-action methods like DQN. TD3 improves on the earlier DDPG algorithm by addressing two known weaknesses — Q-value overestimation and training instability — through three core mechanisms:

1. **Twin critics** — two independently initialized Q-networks; the minimum of the two is used when computing the training target, which counteracts overestimation bias.
2. **Delayed policy updates** — the critic is updated every training step, but the actor (and target networks) are only updated every 2nd step, giving the critic time to stabilize before the policy chases it.
3. **Target policy smoothing** — small, clipped Gaussian noise is added to the target action before it's evaluated, preventing the actor from exploiting narrow, spurious peaks in the critic's value estimates.

## Setup

```bash
# Clone the repo
git clone https://github.com/SaloniChitre/td3-pendulum-swingup.git
cd td3-pendulum-swingup

# Create and activate a virtual environment (conda example)
conda create -n jlab python=3.11
conda activate jlab

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python pendulum.py
```

This will:
1. Train a TD3 agent for 200 episodes on `Pendulum-v1`
2. Print average reward every 10 episodes
3. Save a learning curve plot to `td3_learning_curve.png`
4. Generate a GIF of the trained agent's behavior (`trained_pendulum.gif`), evaluated with zero exploration noise

A full run takes roughly 5–15 minutes on CPU — no GPU required.

## Results

| Metric | Value |
|---|---|
| Random baseline (episodes 1–10) | ~-1,200 to -1,600 reward |
| Converged performance (episodes 190–200) | ~-90 to -170 reward |
| Training episodes | 200 |
| Random exploration warm-up | first 1,000 steps |

See `td3_learning_curve.png` for the full reward-vs-episode curve, and `trained_pendulum.gif` for a visual demonstration of the trained agent swinging up and balancing from a random starting position.

## Implementation details

- **Framework:** PyTorch
- **Network architecture:** 2 hidden layers of 256 units for both actor and critic
- **Actor:** outputs torque via `tanh` activation, scaled to the environment's `[-2, 2]` limit
- **Critic:** two independent Q-networks (twin critics), each taking concatenated state+action as input

### Hyperparameters

All hyperparameters are used as published in Fujimoto et al. (2018), not hand-tuned:

| Parameter | Value | Description |
|---|---|---|
| `discount` (γ) | 0.99 | Bellman equation discount factor |
| `tau` (τ) | 0.005 | Soft update rate for target networks |
| `policy_noise` | 0.2 | Std. dev. of target policy smoothing noise |
| `noise_clip` | 0.5 | Max magnitude of smoothing noise |
| `policy_freq` | 2 | Actor/target update frequency (every N critic updates) |
| `actor_lr` / `critic_lr` | 3e-4 | Adam optimizer learning rates |
| `batch_size` | 256 | Replay buffer sample size |
| `start_timesteps` | 1,000 | Steps of pure random exploration before training begins |
| `buffer capacity` | 1,000,000 | Max replay buffer size |

## Project structure

```
td3-pendulum-swingup/
├── pendulum.py              # Full TD3 implementation, training loop, evaluation
├── requirements.txt         # Python dependencies
├── td3_learning_curve.png   # Generated: reward vs. episode plot
├── trained_pendulum.gif     # Generated: trained agent demonstration
└── README.md
```

## What didn't work (debugging log)

A few real implementation issues encountered and resolved during development:

- **`ReplayBuffer` missing a `__len__` method** — caused a `TypeError` when the training loop checked buffer size (`len(buffer)`); fixed by implementing `__len__` to return the length of the underlying `deque`.
- **Code structure errors from inconsistent indentation** — some classes were accidentally nested inside one another during early drafts; restructured into proper, independent top-level classes: `Actor`, `Critic`, `ReplayBuffer`, `TD3`.
- **Missing `pygame` dependency** — blocked GIF rendering of the trained agent; resolved by installing `gymnasium[classic-control]`.

## AI tool disclosure

AI assistance (Claude) was used for:
- Debugging specific runtime errors (traced above)
- Explaining TD3 theory, the Bellman equation, and each algorithmic component in depth
- Restructuring/formatting assistance

I wrote, ran, and understand every line of the implementation, made the design and hyperparameter decisions, ran the actual training and evaluation, and can explain and defend any part of the code or underlying math.

## References

- Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* (TD3 paper)
- Gymnasium `Pendulum-v1` documentation: https://gymnasium.farama.org/environments/classic_control/pendulum/
