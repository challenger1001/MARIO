import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config import LR, CLIP_EPSILON, GAMMA, GAE_LAMBDA, UPDATE_EPOCHS, BATCH_SIZE, DEVICE
import numpy as np
from config import *


class PPOAgent:
    def __init__(self, model, n_actions):
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.n_actions = n_actions

    def select_action(self, state):
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(DEVICE)
        logits, _ = self.model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_gae(self, rewards, masks, values, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        
        states = torch.from_numpy(np.array(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.array(actions)).to(DEVICE)
        # log_probs  = torch.from_numpy(np.array(log_probs)).float().to(DEVICE)
        # log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(DEVICE)
        returns    = torch.from_numpy(np.array(returns)).float().to(DEVICE)
        advantages = torch.from_numpy(np.array(advantages)).float().to(DEVICE)
        
        # states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        # actions = torch.tensor(actions).to(DEVICE)
        log_probs = torch.tensor(log_probs).to(DEVICE)
        # returns = torch.tensor(returns).to(DEVICE)
        # advantages = torch.tensor(advantages).to(DEVICE)

        for _ in range(UPDATE_EPOCHS):
            logits, values = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
