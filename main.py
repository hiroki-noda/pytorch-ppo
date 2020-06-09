from policy import PPO
from memory import Memory
import gym
import numpy as np
import torch
import argparse
import mujoco_py

def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1000,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 20)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=3000000,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='Reacher-v2',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(args.seed)

    obs = env.reset()
    obs = torch.FloatTensor(obs)
    done = 0.0

    policy = PPO(env.observation_space, 
                env.action_space,
                clip_param=0.2,
                epoch=10,
                num_mini_batch=32,
                value_loss_coef=0.5,
                lr=3e-4,
                eps=1e-5,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss=True)
    memory = Memory(args.num_steps, env.observation_space, env.action_space)

    num_updates = int(
        args.num_env_steps) // args.num_steps
    for j in range(num_updates):
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob = policy.act(obs, done)

            obs, reward, done, infos = env.step(action)
            done = torch.FloatTensor([1.0]) if done else torch.FloatTensor([0.0])
            obs = torch.FloatTensor(obs)
            memory.insert(obs, action, action_log_prob, value, reward, done)
        
        with torch.no_grad():
            next_value = policy.get_value(obs, done)

        memory.compute_returns(next_value, args.gamma)
        policy.update(memory)
        memory.after_update()

if __name__ == "__main__":
    main()
