from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, nn.Sigmoid)

    def forward(self, obs, act):
        q = self.q(th.cat([obs, act], dim=-1))
        return th.squeeze(q, -1)


class EnsemblePolicy(BasePolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            num_nets: int = 5,
    ):


        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        self.features_extractor = self.make_features_extractor()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.num_nets = num_nets
        self.pis = nn.ModuleList([
            mlp([obs_dim] + net_arch + [act_dim], activation_fn, nn.Tanh).to(self.device)
            for _ in range(self.num_nets)
        ])

        self.q1 = MLPQFunction(obs_dim, act_dim, net_arch, activation_fn).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, net_arch, activation_fn).to(self.device)


    def _predict(self, obs: th.Tensor, deterministic: bool = False, i: int = -1) -> th.Tensor:
        features = self.extract_features(obs)
        with th.no_grad():
            if i >= 0:
                return self.pis[i](features)
            actions = th.stack([pi(features) for pi in self.pis], dim=0)
            mean_action = actions.mean(dim=0)
        return mean_action

    def variance(self, obs: np.ndarray) -> float:
        obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        features = self.extract_features(obs)
        with th.no_grad():
            vals = [pi(features).cpu().numpy() for pi in self.pis]
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def safety(self, obs: np.ndarray, act: np.ndarray) -> float:
        # closer to 1 indicates more safe.
        obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        act = th.as_tensor(act, dtype=th.float32, device=self.device)
        features = self.extract_features(obs)
        with th.no_grad():
            return float(th.min(self.q1(features, act), self.q2(features, act)).cpu().numpy())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features =  self.extract_features(obs)
        return self._predict(features)

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        features = super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
        return features