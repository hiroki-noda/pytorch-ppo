from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), 
            nn.Tanh())

        self.critic = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), 
            nn.Tanh())

        self.mean = nn.Linear(hidden_size, action_space.shape[0])
        self.log_std = nn.Linear(hidden_size, action_space.shape[0])

        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, input, done):
        x = input

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        mean = self.mean(hidden_actor)
        log_std = self.log_std(hidden_actor)

        return self.critic_linear(hidden_critic), mean, log_std

class PPO():
    def __init__(self, 
                obs_space, 
                action_space,
                clip_param,
                epoch,
                num_mini_batch,
                value_loss_coef,
                lr=None,
                eps=None,
                max_grad_norm=None,
                use_clipped_value_loss=True):

        self.policy = ActorCritic(obs_space, action_space)
        self.clip_param = clip_param
        self.epoch = epoch
        self.num_mini_batch = num_mini_batch
        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.)

        self.value_loss_coef = value_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)

    def act(self, input, done):
        value, mean, log_std = self.policy(input, done)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        return value, action, log_prob

    def get_value(self, input, done):
        value, mean, _ = self.policy(input, done)
        return value

    def evaluate(self, input, done, action):
        value, mean, log_std = self.policy(input, done)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        entropy = normal.entropy()
        return value, log_prob, entropy

    def update(self, memory):
        advantages = memory.returns[:-1] - memory.value_preds[:-1]
        (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for e in range(self.epoch):
            data = memory.generator(advantages, self.num_mini_batch)
            for sample in data:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample
                values, action_log_probs, dist_entropy = self.evaluate(obs_batch, masks_batch, actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (return_batch - values).pow(2).mean()
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        num_updates = self.epoch * self.num_mini_batch
