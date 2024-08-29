import torch
import torch.nn as nn
import torch.optim as optim

class A2C(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(A2C, self).__init__()
        # conv and fc for actor critic
        self.actor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):

        action_prob = self.actor(x.view(x.size(0), 1, 84, 84))
        state_value = self.critic(x.view(x.size(0), 1, 84, 84))
        return action_prob, state_value

    def select_action(self, state):
        action_prob, _ = self.forward(state)
        action = torch.multinomial(action_prob, 1)
        return action.squeeze()

    def compute_loss(self, state, action, reward, next_state, done):
        action_prob, state_value = self.forward(state)
        _, next_state_value = self.forward(next_state)

        log_prob = torch.log(action_prob[0, action] + 1e-10)
        advantage = reward + (1 - done) * 0.99 * next_state_value - state_value

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        return actor_loss + 0.5 * critic_loss

    def update(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class mixnet(nn.Module):
