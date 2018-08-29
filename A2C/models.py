import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActionNetwork(nn.Module):
    def __init__(self, state_dim, action_num, action_dim, hidden_dim, n_layers):
        super().__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        ]
        for i in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]

        self.feature_extractor = nn.Sequential(*layers)
        self.action_output = nn.Linear(hidden_dim, action_num * action_dim)

    def forward(self, state):
        features = self.feature_extractor(state)
        action_log_prob = F.log_softmax(self.action_output(features).view(-1, self.action_num, self.action_dim), 2)
        return action_log_prob


class ValueNetwork(nn.Module):
    def __init__(self, state_dim,  hidden_dim, n_layers):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        ]
        for i in range(n_layers-1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]

        self.feature_extractor = nn.Sequential(*layers)
        self.value_output = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.value_output(features)
        return value


class CheckPoint:
    def __init__(self, env, state):
        self.env = env
        self.state = state


class A2CAgent:
    def __init__(self, state_dim, action_num, action_dim,
                 hidden_dim=32, n_layers=1,
                 lr=0.01, lr_decay=0,
                 gamma=0.99,
                 render_learning=False):

        self.state_dim = state_dim
        self.action_num = action_num
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gamma = gamma

        self.actor = ActionNetwork(state_dim, action_num, action_dim, hidden_dim, n_layers)
        self.critic = ValueNetwork(state_dim, hidden_dim, n_layers)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=lr_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=lr_decay)

        self.render_learning = render_learning

    def rollout(self, check_point,  sample_step=32):
        env = check_point.env
        state = check_point.state

        step_counter = 0
        done = False

        state_list = []
        reward_list = []
        action_list = []

        final_value = 0

        # generate samples
        while step_counter < sample_step and not done:
            if self.render_learning:
                env.render()
            action_log_prob = self.actor(torch.Tensor(state).float().view(1, -1))
            action_prob = torch.exp(action_log_prob)
            action = torch.multinomial(action_prob.squeeze(), 1).view(-1).numpy().tolist()

            if len(action) == 1:
                action = action[0]

            next_state, reward, done, info = env.step(action)

            state_list += [state.reshape(1, -1)]
            action_list += [action]
            reward_list += [reward]

            state = next_state
            step_counter += 1

        if done:
            check_point.state = env.reset()
        else:
            check_point.state = state
            final_value = self.critic(torch.Tensor(state).float().view(1, -1)).view(-1).item()

        # calculate discounted reward reversely
        q_value_list = np.zeros(len(reward_list))
        for i in reversed(range(len(reward_list))):
            final_value = reward_list[i] + final_value * self.gamma
            q_value_list[i] = final_value

        state_list = np.concatenate(state_list, 0)
        action_list = np.asarray(action_list).reshape(-1, self.action_num, 1)
        q_value_list = q_value_list.reshape(-1, 1)

        return (state_list, action_list, q_value_list), check_point

    def learn(self, state_list, action_list, q_value_list):
        states = torch.from_numpy(state_list).float()
        actions = torch.from_numpy(action_list).long()
        q_values = torch.from_numpy(q_value_list).float()

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        values = self.critic(states)
        critic_loss = F.mse_loss(values, q_values)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        advantages = q_values - values
        log_probs = self.actor(states).gather(2, actions)
        actor_loss = (log_probs.sum(dim=1) * advantages.detach()).mean() * -1
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def evaluate(self, env, n_episode=10, max_step=500):
        total_reward = 0
        for i in range(n_episode):
            state = env.reset()
            step = 0
            done = False

            while step < max_step and not done:
                action_log_prob = self.actor(torch.Tensor(state).float().view(1, -1))
                action_prob = torch.exp(action_log_prob)
                action = torch.multinomial(action_prob.squeeze(), 1).view(-1).numpy().tolist()

                if len(action) == 1:
                    action = action[0]

                next_state, reward, done, info = env.step(action)

                state = next_state
                total_reward += reward
                step += 1

        return total_reward / n_episode
