# td3-pendulum-swingup
# TD3 for Inverted Pendulum Swing-Up

Solution to Jefferson Lab's Interview Problem #1 — solving the Pendulum-v1
swing-up task using the TD3 (Twin Delayed DDPG) algorithm.

## Setup
\`\`\`bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`

## Run
\`\`\`bash
python pendulum.py
\`\`\`

## Results
- Final test performance: reward converges from ~-1200 to -1600 (random baseline)
  to ~-90 to -170 after 200 episodes.
- See `td3_learning_curve.png` for the learning curve and `trained_pendulum.gif`
  for the trained agent in action.

## Implementation notes
- Hyperparameters used as published in Fujimoto et al., 2018 (TD3 paper),
  not hand-tuned.
- AI assistance was used for debugging and understanding the underlying math;
  the implementation, training, and evaluation are my own.
