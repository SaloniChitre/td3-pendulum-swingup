#Importing torch libraries
import torch
import torch.nn as nn #for linearity
import torch.nn.functional as F #activation function i.r reLU and tanh

import numpy as np #To handle double dimensional arrays outside of the pytorch tensors
import gymnasium as gym #Gymnasium library to handle the physics of the pendulum
from collections import deque #replay buffer to drop old entries once it reaches max limit
import random #for random sampling batches

#---ACTOR---

class Actor(nn.Module): #nn.module for inheritance of the pytorch modules including parameter tuning
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()           #to run nn.module's own setup code first before adding tha layers

        #3 fully connected layers
        self.l1 = nn.Linear(state_dim, 256) # takes 3 inputs , gives 256 outputs(weighted sum of all 3 inputs+bias)
        self.l2 = nn.Linear(256, 256) #remains 256 to 256
        self.l3 = nn.Linear(256, action_dim) #action_dim is 1 because it narrows down to one torque value to the output

        self.max_action = max_action #storing 2.0 for pendulum to use in forward()

    def forward(self, state):  #to define what happens when you call the network on an input
        a = F.relu(self.l1(state)) #to push the state through the first linear layer
        a = F.relu(self.l2(a)) #for second layer
        return self.max_action * torch.tanh(self.l3(a)) #tanh- (-1,1) , max_action(2.0-actual pendulum torque limits therefore range is (-2,2))


#----CRITIC----

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)  #To evaluate state+action for estimating the Q-value
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)   # 1 scalar estimate of output action  

        # Q2 (the twin)
        self.l4 = nn.Linear(state_dim + action_dim, 256) #same architecture but different network so that the errors of Q1 wont correlate
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action): #concatenates state and action as required to calculate the Q-value
        sa = torch.cat([state, action], dim=1)

        #pushing the combined inputs through both networks independantly, then return both estimates  
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    #convinience method to compute first critics output to calculate the actors gradient
    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1))
        return self.l3(q1)


#Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=1_000_000): #max length is 1 million, then deques
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) #picks batch_size unique random items from the buffer
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) #all states together, all actions together, etc., then stacks each one of those lists into a numpy array

        #converting everything to pytorch tensors as the network expects
        return (torch.FloatTensor(state), torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state), torch.FloatTensor(done).unsqueeze(1))

    def __len__(self): #tells python how to compute len(buffer), just returns the length of the underlying deque
        return len(self.buffer)


#----TD3 AGENT----

class TD3:
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, actor_lr=3e-4, critic_lr=3e-4):

        #for actor
        self.actor = Actor(state_dim, action_dim, max_action) #Main actor
        self.actor_target = Actor(state_dim, action_dim, max_action) #Target actor
        self.actor_target.load_state_dict(self.actor.state_dict()) #load_state_dict() loads learned parameter into another network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr) #Adam is the optimizer that updates weights based on the gradient descend 

        #for critic 
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        #Values used from fujimotos 2018 paper because that is the standard practice that works while training an agent, I tried using 
        #different values

        self.max_action = max_action #2.0 for pendulum v1
        self.discount = discount  #gamma=0.99 in the bellman equation so that agent has motivation to keep learning
        self.tau = tau #how fast the target network track the main network during soft update = 0.005 , each update 0.5% differnt from main network
        self.policy_noise = policy_noise #standard deviation of noise added to the target action = 0.2
        self.noise_clip = noise_clip #noise_clip=0.5 limit so any random bigger number doesnt corrupt the training step
        self.policy_freq = policy_freq #policy_freq=2 i.e critic updates every training step but actor updates every 2nd training step
        self.total_it = 0 #counts how many times train() is being called

    def select_action(self, state, noise=0.1): #.reshape(1, -1) turns a flat state array (shape (3,)) into a batch of size 1 (shape (1, 3)), because PyTorch layers expect batched input even when you're only passing one example
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().numpy().flatten() #running the state through the actor , .detach() -disconnect o/p from pytorch gradient tracking graph

        if noise != 0:
            action += np.random.normal(0, noise * self.max_action, size=action.shape) #only used during traning, while evaluating trained actions noise = 0

        return action.clip(-self.max_action, self.max_action) #clip in the [-2,2] range again

    #---TRAINING---

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1 #policy_freq counter

        state, action, reward, next_state, done = replay_buffer.sample(batch_size) #pull random batch of 256 past experiences

        with torch.no_grad(): #to calculate the target Q-value to regress toward
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) #.clamp() to hard-caps it to [-.5,0.5] so random extreme should not distort the actors next action
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action) #target policy smoothning

            target_Q1, target_Q2 = self.critic_target(next_state, next_action) #Both target critics evaluate the (next_state, next_action) pair,we take the smaller of the two estimates
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q #The Bellman target: immediate reward, plus discounted future value (zeroed out if the episode ended)

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #training without identical critic copies

        #clear old gradients (they accumulate by default, so you must reset them each step), compute new gradients via backpropagation, then update the weights using those gradients

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Twin delay
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean() #Feed the current states through the current actor to get actions, then ask the critic how good those actions are. Negate it (since we want to maximize Q but optimizers minimize), and .mean() averages the loss across the batch into a single number to backpropagate from

            #same for actors
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            #---Crtics----

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            #----Actor----

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        #.item() extracts a plain Python float from a single-element tensor — useful for logging/printing. Returns None for actor_loss on steps where the actor wasn't updated
        return critic_loss.item(), (actor_loss.item() if actor_loss is not None else None)


#Physics of the pendulum
def train_td3(episodes=200, batch_size=256, start_timesteps=1000):
    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0] #cos/sin value to avoid confusing the angle of the pendulum as angles are between 0-360 degree

    action_dim = env.action_space.shape[0] # action_dim=1
    max_action = float(env.action_space.high[0]) # max_action=2

    #Setting up the agent, an empty buffer, and tracking lists/counters
    agent = TD3(state_dim, action_dim, max_action)
    buffer = ReplayBuffer()
    episode_rewards = []
    total_steps = 0

    for ep in range(episodes):  #resets or starts a new episode for the pendulum, returns (state,info) we ignore the info with an _
        state, _ = env.reset()
        ep_reward = 0
        done = False
        truncated = False

        while not (done or truncated): #Pure random exploration for the first 1000 steps total (not per episode), then switch to the agent's noisy policy
            if total_steps < start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=0.1)

            next_state, reward, done, truncated, _ = env.step(action) #given the torque we chose, the environment computes the pendulum's new angle/velocity and the reward for this step
            buffer.add(state, action, reward, next_state, float(done)) #storing the transition,float(done) converts the boolean to 0.0 or 1.0 so it can be used directly in the Bellman equation math 

            #advance to the next state
            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) > batch_size and total_steps > start_timesteps:
                agent.train(buffer, batch_size)

        episode_rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {ep+1}, Avg Reward (last 10): {avg:.2f}")

    return agent, episode_rewards
if __name__ == "__main__":
    agent, rewards = train_td3(episodes=200)

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("TD3 on Pendulum-v1: Learning Curve")
    plt.savefig("td3_learning_curve.png")
    plt.show()

    # --- Generate GIF of trained agent ---
    import imageio

    eval_env = gym.make("Pendulum-v1", render_mode="rgb_array") #separate env just for evaluation, keeps it independent from training
    state, _ = eval_env.reset()
    frames = []

    for _ in range(200):
        action = agent.select_action(state, noise=0)  # noise=0: show the pure trained policy, no exploration randomness
        state, reward, done, truncated, _ = eval_env.step(action)
        frames.append(eval_env.render())
        if done or truncated:
            break

    imageio.mimsave("trained_pendulum.gif", frames, fps=30)
    print("Saved trained_pendulum.gif")