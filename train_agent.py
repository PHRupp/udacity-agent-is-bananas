from collections import deque
from typing import List

import numpy as np
import torch

from base_agent import BaseAgent
from utils import logger


def train(
    env,
    agent: BaseAgent,
    n_episodes: int = 2000,
    max_t: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
) -> List[float]:
    """Deep Q-Learning.

    Params
    ======
        env: environment that agent will train in
        agent (BaseAgent): agent to be trained
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores: List[float] = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    # loop through each episode
    for i_episode in range(1, n_episodes + 1):
        logger.debug('Episode %d', i_episode)
        state_brain = env.reset()
        state = state_brain['BananaBrain'].__dict__['vector_observations']
        score = 0

        # loop through all time steps within episode
        for t in range(max_t):
            action = agent.act(state, eps)
            logger.debug('Episode %d, Time %d, Chosen Action %d', i_episode, t, action)
            state_brain = env.step(vector_action=int(action))
            next_state = state_brain['BananaBrain'].__dict__['vector_observations'][0]
            reward = state_brain['BananaBrain'].__dict__['rewards'][0]
            done = state_brain['BananaBrain'].__dict__['local_done'][0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            # exit the episode if done condition reached
            if done:
                break

        # save most recent score
        scores_window.append(score)
        scores.append(score)

        # decrease epsilon
        eps = max(eps_end, eps_decay * eps)

        score_str = '\rEpisode {}\tAverage Score: {:.2f}'

        logger.info(score_str.format(i_episode, np.mean(scores_window)))
        if (i_episode % 100) == 0:
            logger.info(score_str.format(i_episode, np.mean(scores_window)))

        # If the avg score of latest window is above threshold, then stop training and save model
        if np.mean(scores_window) >= 10.0:
            solved_str = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
            logger.info(solved_str.format(i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores
