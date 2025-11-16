import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from gait_wrapper import GaitWrapper


env = env = GaitWrapper(gym.make("BipedalWalker-v3", hardcore=False, render_mode=None))


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu(x)
        std = torch.exp(self.log_std).clamp(min=1e-3)

        return mu, std
    

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.fc(state)
    

def compute_gae(values, rewards, gamma=0.99, lam=0.95):
    values = np.array(values)
    rewards = np.array(rewards)
    advantages = np.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae

        advantages[t] = gae
    
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def gait_reward(obs, phase_left, phase_right, d_lower=-0.4):
    left_contact  = obs[8]
    right_contact = obs[9]

    reward = 0.0

    # stance when phase >= d_lower
    if phase_left >= d_lower and left_contact == 0:
        reward -= 0.5

    # swing when phase <= -d_lower
    if phase_left <= -d_lower and left_contact == 1:
        reward -= 0.5

    # same for right
    if phase_right >= d_lower and right_contact == 0:
        reward -= 0.5

    if phase_right <= -d_lower and right_contact == 1:
        reward -= 0.5

    return reward



state_dim = 26
action_dim = env.action_space.shape[0]

actor = PolicyNetwork(state_dim, action_dim)
critic = ValueNetwork(state_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

n_steps = 2000
batch_size = 64
n_epochs = 10
gamma = 0.99
lam = 0.95
eps_clip = 0.2
n_episodes = 1000

state = env.reset()[0]


for episode in range(n_episodes):
    states, actions, rewards, log_probs, values = [], [], [], [], []

    for _ in range(n_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        mu, std = actor(state_tensor)
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy()[0])

        value = critic(state_tensor).item()
        reward += gait_reward(next_state, info["phase_left"], info["phase_right"])

        states.append(state)
        actions.append(raw_action.detach().numpy()[0])
        log_probs.append(log_prob.item())
        rewards.append(reward)
        values.append(value)

        state = next_state
        if terminated or truncated:
            state = env.reset()[0]

    advantages, returns = compute_gae(values, rewards)

    states = torch.FloatTensor(np.array(states))
    actions = torch.FloatTensor(np.array(actions))
    old_log_probs = torch.FloatTensor(np.array(log_probs))
    returns = torch.FloatTensor(np.array(returns))
    advantages = torch.FloatTensor(np.array(advantages))

    for _ in range(n_epochs):
        indices = torch.randperm(n_steps)

        for start in range(0, n_steps, batch_size):
            batch_idx = indices[start:start + batch_size]

            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]
            adv_batch = advantages[batch_idx]
            ret_batch = returns[batch_idx]
            old_lp_batch = old_log_probs[batch_idx]

            mu, std = actor(s_batch)
            dist = Normal(mu, std)
            scaled_actions = torch.tanh(a_batch)
            log_probs_new = dist.log_prob(a_batch) - torch.log(1 - scaled_actions.pow(2) + 1e-6)
            log_probs_new = log_probs_new.sum(dim=-1)
            ratio = torch.exp(log_probs_new - old_lp_batch)

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_batch

            actor_loss = -torch.min(surr1, surr2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            value_preds = critic(s_batch).squeeze()
            critic_loss = (ret_batch - value_preds).pow(2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

    if episode % 10 == 0:
        print(f"Episode: {episode}, reward: {sum(rewards)}")



env = GaitWrapper(gym.make("BipedalWalker-v3", hardcore=False, render_mode="human"))

for episode in range(10):
    state = env.reset()[0]
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            mu, std = actor(state_tensor)
            dist = Normal(mu, std)
            action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
        done = terminated or truncated
        state = next_state
