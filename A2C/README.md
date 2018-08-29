(Synchronous) Advantage Actor-Critic
===============

This is a naive implementation of synchronous version of Advantage Actor-Critic.
The original idea can be found in [
Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).

## Requirements

    * gym
    * numpy
    * torch==0.4.0

## User Manual

```
$ python run_a2c.py --help
```
and you will see the following help message.

```
usage: run_a2c.py [-h] [--n_process p] [--max_learning_step m]
                  [--hidden_dim h] [--n_layers l] [--gamma g] [--lr lr]
                  [--evaluate_interval ei] [--evaluate_episode ee]
                  [--evaluate_max_step ems]
                  env_name state_dim action_num action_dim

positional arguments:
  env_name              OpenAI Gym environment name, e.g. "CartPole-v0"
  state_dim             Dimension of state vector of environment
  action_num            Number of action that agent can take in each step
  action_dim            Dimension of action vector of environment

optional arguments:
  -h, --help            show this help message and exit
  --n_process p         Number of process to launch in A2C, 1 means no multi-
                        processing
  --max_learning_step m
                        Max number of steps agent learns
  --hidden_dim h        Dimension of hidden layers in MLP
  --n_layers l          Number of hidden layers in MLP
  --gamma g             Discount factor for future reward
  --lr lr               Learning rate
  --evaluate_interval ei
                        Evaluate agent every ei steps
  --evaluate_episode ee
                        Evaluate ee episode and take average
  --evaluate_max_step ems
                        Run ems steps in each episode for evaluation

```

## Examples

* Play [Cartpole](https://gym.openai.com/envs/CartPole-v0/)

```
$ python run_a2c.py CartPole-v0 4 1 2
```




