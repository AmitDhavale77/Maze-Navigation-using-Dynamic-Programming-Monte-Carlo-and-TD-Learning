# Maze Navigation using Dynamic Programming, Monte Carlo, and TD Learning

## Overview

This project aims to solve a maze environment modelled as a **Markov Decision Process (MDP)**, where an agent navigates the maze to find the optimal policy to maximize rewards. The agent can take actions such as moving north, east, south, or west, with a success probability *p*. The goal is to apply reinforcement learning techniques like **Dynamic Programming**, **Monte Carlo**, and **Temporal Difference (TD) Learning** to find the optimal policy without full knowledge of the environment's dynamics (i.e., transition matrix **T** and reward function **R**).

The maze environment includes obstacles (black squares), absorbing states (dark-grey squares), and a random starting position for each episode. The agent's goal is to reach an absorbing state while maximizing rewards, and the episode ends when the agent reaches one of the absorbing states or takes more than 500 steps.

## Environment Description

The maze is a grid where:

- **Black squares** represent obstacles that the agent cannot pass through.
- **Dark-grey squares** represent absorbing (terminal) states that give specific rewards.
- The agent starts at a random state (S) at the beginning of each episode.
- The agent can perform four possible actions at each time step:
  - **a0**: Move north
  - **a1**: Move east
  - **a2**: Move south
  - **a3**: Move west

The action chosen has a probability *p* of succeeding, and if it fails, the agent moves in one of the other three directions with equal probability. If the agent attempts to move into an obstacle, it remains in the current position but can move in other directions with reduced probability.

### Reward Function

- **Reward for each action**: The agent receives a reward of **-1** for each action, except when reaching an absorbing state where it receives a terminal reward.
- **Terminal States**: Absorbing states are terminal states with no transition to any other state.

### Parameters

- **p**: The probability of the agent's action succeeding (given as a hyperparameter).
- **Starting States**: Randomly selected at the start of each episode, from predefined possible start states.

### Episode Termination

- The episode ends when the agent reaches one of the absorbing states.
- The episode also terminates after 500 steps if the agent has not reached a terminal state.

## Reinforcement Learning Techniques

### 1. **Dynamic Programming (DP)**

Dynamic Programming is used to find the optimal policy and value function by leveraging the full knowledge of the environment. This approach will involve iterating over all states in the maze, using methods like **Value Iteration** or **Policy Iteration** to compute the optimal policy.

### 2. **Monte Carlo (MC) Learning**

Monte Carlo methods rely on sampling episodes of the environment to estimate the value of each state or action without requiring knowledge of the environment's dynamics. This technique involves running episodes and updating the value of states based on the return (cumulative reward) received.

### 3. **Temporal-Difference (TD) Learning**

TD learning combines ideas from both Monte Carlo and Dynamic Programming methods. It updates value estimates based on the current state-action pair and the next state, without requiring a model of the environment's transition matrix.
