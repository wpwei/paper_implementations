import gym
from A2C.models import A2CAgent, CheckPoint
from multiprocessing import Pool
import numpy as np
import argparse


def launch(env_name, n_process, max_learning_step, evaluate_interval,
           state_dim, action_num, action_dim, hidden_dim, n_layers, gamma, lr,
           evaluate_episode, evaluate_max_step,
           render_learning=False):

    check_points = []
    for i in range(n_process):
        env = gym.make(env_name)
        state = env.reset()
        check_points += [CheckPoint(env, state)]

    agent = A2CAgent(state_dim=state_dim, action_num=action_num, action_dim=action_dim,
                     hidden_dim=hidden_dim, n_layers=n_layers, gamma=gamma,
                     lr=lr, render_learning=render_learning)

    if n_process > 1:
        pool = Pool()

        for i in range(max_learning_step):
            [samples, check_points] = list(zip(*pool.map(agent.rollout, check_points)))
            samples = [np.concatenate(s, 0) for s in list(zip(*samples))]
            agent.learn(*samples)
            if i % evaluate_interval == 0:
                print(f'Step {i}: {agent.evaluate(gym.make(env_name), n_episode=evaluate_episode, max_step=evaluate_max_step)}')

    elif n_process == 1:
        check_point = check_points[0]
        for i in range(max_learning_step):
            sample, check_point = agent.rollout(check_point)
            agent.learn(*sample)
            if i % evaluate_interval == 0:
                print(f'Step {i}: {agent.evaluate(gym.make(env_name), n_episode=evaluate_episode, max_step=evaluate_max_step)}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('env_name',
                        help='OpenAI Gym environment name, e.g. \"CartPole-v0\"', type=str)

    parser.add_argument('state_dim',
                        help='Dimension of state vector of environment', type=int)

    parser.add_argument('action_num',
                        help='Number of action that agent can take in each step', type=int)

    parser.add_argument('action_dim',
                        help='Dimension of action vector of environment', type=int)

    parser.add_argument('--n_process', default=1, metavar='p',
                        help='Number of process to launch in A2C, 1 means no multi-processing', type=int)

    parser.add_argument('--max_learning_step', default=2000, metavar='m',
                        help='Max number of steps agent learns', type=int)

    parser.add_argument('--hidden_dim', default=32, metavar='h',
                        help='Dimension of hidden layers in MLP', type=int)

    parser.add_argument('--n_layers', default=2, metavar='l',
                        help='Number of hidden layers in MLP', type=int)

    parser.add_argument('--gamma', default=0.99, metavar='g',
                        help='Discount factor for future reward', type=float)

    parser.add_argument('--lr', default=0.01, metavar='lr',
                        help='Learning rate', type=int)

    parser.add_argument('--evaluate_interval', default=50, metavar='ei',
                        help='Evaluate agent every ei steps', type=int)

    parser.add_argument('--evaluate_episode', default=1, metavar='ee',
                        help='Evaluate ee episode and take average', type=int)

    parser.add_argument('--evaluate_max_step', default=500, metavar='ems',
                        help='Run ems steps in each episode for evaluation', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    launch(
        args.env_name,
        args.n_process,
        args.max_learning_step,
        args.evaluate_interval,
        args.state_dim,
        args.action_num,
        args.action_dim,
        args.hidden_dim,
        args.n_layers,
        args.gamma,
        args.lr,
        args.evaluate_episode,
        args.evaluate_max_step,
    )
