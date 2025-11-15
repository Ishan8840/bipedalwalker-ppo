import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from gymnasium.vector import SyncVectorEnv


num_envs = 4  # number of parallel envsironments

def make_env():
    return gym.make("BipedalWalker-v3", render_mode=None)

envs = SyncVectorEnv([make_env for _ in range(num_envs)])

states = envs.reset()[0]


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


state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

actor = PolicyNetwork(state_dim, action_dim)
critic = ValueNetwork(state_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

n_steps = 1024
batch_size = 64
n_epochs = 10
gamma = 0.99
lam = 0.95
eps_clip = 0.2
n_episodes = 500

states= envs.reset()[0]


for episode in range(n_episodes):
    mb_states, mb_actions, mb_rewards, mb_log_probs, mb_values = [], [], [], [], []

    for stp in range(n_steps):
        state_tensor = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            mu, std = actor(state_tensor)
            dist = Normal(mu, std)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            values = critic(state_tensor).squeeze()


        next_states, rewards, terminated, truncated, _ = envs.step(actions.numpy())
        done = np.logical_or(terminated, truncated)        

        mb_states.append(states.copy())
        mb_actions.append(actions.numpy())
        mb_rewards.append(rewards)
        mb_log_probs.append(log_probs.numpy())
        mb_values.append(values.numpy())

        states = next_states

        # Reset done environments
        for i, d in enumerate(done):
            if d:
                states[i] = envs.reset()[0][i]


    mb_states = np.array(mb_states).reshape(-1, state_dim)
    mb_actions = np.array(mb_actions).reshape(-1, action_dim)
    mb_rewards = np.array(mb_rewards).reshape(-1)
    mb_log_probs = np.array(mb_log_probs).reshape(-1)
    mb_values = np.array(mb_values).reshape(-1)

    # Compute GAE
    advantages, returns = compute_gae(mb_values, mb_rewards, gamma, lam)

    states_tensor = torch.tensor(mb_states, dtype=torch.float32)
    actions_tensor = torch.tensor(mb_actions, dtype=torch.float32)
    old_log_probs_tensor = torch.tensor(mb_log_probs, dtype=torch.float32)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

    n_samples = states_tensor.shape[0]
    for _ in range(n_epochs):
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start+batch_size]

            s_batch = states_tensor[batch_idx]
            a_batch = actions_tensor[batch_idx]
            adv_batch = advantages_tensor[batch_idx]
            ret_batch = returns_tensor[batch_idx]
            old_lp_batch = old_log_probs_tensor[batch_idx]

            mu, std = actor(s_batch)
            dist = Normal(mu, std)
            log_probs_new = dist.log_prob(a_batch).sum(dim=-1)
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

    if episode % 5 == 0:
        print(f"Episode: {episode}, reward: {mb_rewards.mean():.2f}")