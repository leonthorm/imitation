import glob
import os
import platform
import random
import re
from collections import deque
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cloudpickle
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit


def check_for_correct_spaces_multi_robot(
        env: GymEnv, observation_space: spaces.Space, action_space: spaces.Space, n_robots: int
) -> None:
    """
    Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space

    :param env: Environment to check for valid spaces
    :param observation_space: Observation space to check against
    :param action_space: Action space to check against
    :param n_robots: Number of robots
    """
    observation_space_shape  = (n_robots, observation_space.shape)
    action_space_shape = n_robots * action_space.shape
    if observation_space_shape != env.observation_space.shape:
        raise ValueError(f"Observation spaces shapes do not match: {observation_space_shape} != {env.observation_space.shape}")
    if action_space_shape != env.action_space.shape:
        raise ValueError(f"Action spaces shapes do not match: {action_space_shape} != {env.action_space.shape}")