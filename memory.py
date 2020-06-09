import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Memory(object):
    def __init__(self, num_steps, obs_space, action_space):
        obs_shape = obs_space.shape[0]
        action_shape = action_space.shape[0]
        self.obs = torch.zeros(num_steps + 1, obs_shape)
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        self.actions = torch.zeros(num_steps, action_shape)
        self.masks = torch.ones(num_steps + 1, 1)
        self.step = 0
        self.num_steps = num_steps
        
    def push(self, obs, action, action_log_prob, value_pred, reward, mask):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred.squeeze())
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            # R(t) = γ*R(t+1) + r(t)
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        batch_size = self.rewards.size()[0]
        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, self.obs.size()[1])[indices]
            actions_batch = self.actions.view(-1, self.actions.size()[-1])[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ