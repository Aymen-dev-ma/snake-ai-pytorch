import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.actor = Linear_QNet(input_size, hidden_size, output_size)
        self.critic = Linear_QNet(input_size, hidden_size, 1)

    def forward(self, x):
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOTrainer:
    def __init__(self, model, lr, gamma, eps_clip):
        self.model = model
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, memory):
        states = torch.tensor([m[0] for m in memory], dtype=torch.float)
        actions = torch.tensor([m[1] for m in memory], dtype=torch.long)
        rewards = torch.tensor([m[2] for m in memory], dtype=torch.float)
        next_states = torch.tensor([m[3] for m in memory], dtype=torch.float)
        dones = torch.tensor([m[4] for m in memory], dtype=torch.float)

        old_probs, old_values = self.model(states)
        old_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        returns = []
        Gt = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            Gt = reward + (self.gamma * Gt * (1 - done))
            returns.insert(0, Gt)

        returns = torch.tensor(returns, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(10):  # Update policy 10 times per step
            new_probs, new_values = self.model(states)
            new_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = new_probs / old_probs
            advantage = returns - old_values.detach().squeeze()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.criterion(new_values.squeeze(), returns) - 0.01 * (new_probs * torch.log(new_probs)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


