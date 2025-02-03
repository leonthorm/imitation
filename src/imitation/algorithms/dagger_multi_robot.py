"""DAgger (https://arxiv.org/pdf/1011.0686.pdf).

Interactively trains policy by collecting some demonstrations, doing BC, collecting more
demonstrations, doing BC again, etc. Initially the demonstrations just come from the
expert's policy; over time, they shift to be drawn more and more from the imitator's
policy.
"""

import abc
import logging
import os
import pathlib
import uuid
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from torch.utils import data as th_data
from gymnasium.spaces import Box, Space

from imitation.algorithms import base, bc_multi_robot
from imitation.data import rollout_multi_robot, serialize, types
from imitation.util import logger as imit_logger
from imitation.util import util


class BetaSchedule(abc.ABC):
    """Computes beta (% of time demonstration action used) from training round."""

    @abc.abstractmethod
    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round.

        Args:
            round_num: the current round number. Rounds are assumed to be sequentially
                numbered from 0.

        Returns:
            The fraction of the time to sample a demonstrator action. Robot
                actions will be sampled the remainder of the time.
        """  # noqa: DAR202


class LinearBetaSchedule(BetaSchedule):
    """Linearly-decreasing schedule for beta."""

    def __init__(self, rampdown_rounds: int) -> None:
        """Builds LinearBetaSchedule.

        Args:
            rampdown_rounds: number of rounds over which to anneal beta.
        """
        self.rampdown_rounds = rampdown_rounds

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta linearly decreasing from `1` to `0` between round `0` and
            `self.rampdown_rounds`. After that, it is 0.
        """
        assert round_num >= 0
        return min(1, max(0, (self.rampdown_rounds - round_num) / self.rampdown_rounds))


class ExponentialBetaSchedule(BetaSchedule):
    """Exponentially decaying schedule for beta."""

    def __init__(self, decay_probability: float):
        """Builds ExponentialBetaSchedule.

        Args:
            decay_probability: the decay factor for beta.

        Raises:
            ValueError: if `decay_probability` not within (0, 1].
        """
        if not (0 < decay_probability <= 1):
            raise ValueError("decay_probability lies outside the range (0, 1].")
        self.decay_probability = decay_probability

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta as `self.decay_probability ^ round_num`
        """
        assert round_num >= 0
        return self.decay_probability ** round_num


def reconstruct_trainer(
        scratch_dir: types.AnyPath,
        venv: vec_env.VecEnv,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        device: Union[th.device, str] = "auto",
) -> "DAggerTrainer":
    """Reconstruct trainer from the latest snapshot in some working directory.

    Requires vectorized environment and (optionally) a logger, as these objects
    cannot be serialized.

    Args:
        scratch_dir: path to the working directory created by a previous run of
            this algorithm. The directory should contain `checkpoint-latest.pt` and
            `policy-latest.pt` files.
        venv: Vectorized training environment.
        custom_logger: Where to log to; if None (default), creates a new logger.
        device: device on which to load the trainer.

    Returns:
        A deserialized `DAggerTrainer`.
    """
    custom_logger = custom_logger or imit_logger.configure()
    scratch_dir = util.parse_path(scratch_dir)
    checkpoint_path = scratch_dir / "checkpoint-latest.pt"
    trainer = th.load(checkpoint_path, map_location=utils.get_device(device))
    trainer.venv = venv
    trainer._logger = custom_logger
    return trainer


def _save_dagger_demo(
        trajectory: types.Trajectory,
        trajectory_index: int,
        save_dir: types.AnyPath,
        rng: np.random.Generator,
        prefix: str = "",
) -> None:
    save_dir = util.parse_path(save_dir)
    assert isinstance(trajectory, types.Trajectory)
    actual_prefix = f"{prefix}-" if prefix else ""
    randbits = int.from_bytes(rng.bytes(16), "big")
    random_uuid = uuid.UUID(int=randbits, version=4).hex
    filename = f"{actual_prefix}dagger-demo-{trajectory_index}-{random_uuid}.npz"
    npz_path = save_dir / filename
    assert (
        not npz_path.exists()
    ), "The following DAgger demonstration path already exists: {0}".format(npz_path)
    serialize.save(npz_path, [trajectory])
    logging.info(f"Saved demo at '{npz_path}'")


class InteractiveTrajectoryCollectorMultiRobot(vec_env.VecEnvWrapper):
    """DAgger VecEnvWrapper for querying and saving expert actions.

    Every call to `.step(actions)` accepts and saves expert actions to `self.save_dir`,
    but only forwards expert actions to the wrapped VecEnv with probability
    `self.beta`. With probability `1 - self.beta`, a "robot" action (i.e
    an action from the imitation policy) is forwarded instead.

    Demonstrations are saved as `TrajectoryWithRew` to `self.save_dir` at the end
    of every episode.
    """

    traj_accum: Optional[rollout_multi_robot.TrajectoryAccumulator]
    _last_obs: Optional[np.ndarray]
    _last_user_actions: Optional[np.ndarray]

    def __init__(
            self,
            venv: vec_env.VecEnv,
            get_robot_acts: Callable[[np.ndarray], np.ndarray],
            beta: float,
            save_dir: types.AnyPath,
            rng: np.random.Generator,
            n_robots: int,
            actions_size_single_robot: int
    ) -> None:
        """Builds InteractiveTrajectoryCollector.

        Args:
            venv: vectorized environment to sample trajectories from.
            get_robot_acts: get robot actions that can be substituted for
                human actions. Takes a vector of observations as input & returns a
                vector of actions.
            beta: fraction of the time to use action given to .step() instead of
                robot action. The choice of robot or human action is independently
                randomized for each individual `Env` at every timestep.
            save_dir: directory to save collected trajectories in.
            rng: random state for random number generation.
            n_robots: number of robots for which to sample trajectories. Each robot
                needs to have the same action and observation space.
        """
        super().__init__(venv)
        self.get_robot_acts = get_robot_acts
        assert 0 <= beta <= 1
        self.beta = beta
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None
        self.rng = rng
        self.n_robots = n_robots
        self.actions_size_single_robot = actions_size_single_robot

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set the seed for the DAgger random number generator and wrapped VecEnv.

        The DAgger RNG is used along with `self.beta` to determine whether the expert
        or robot action is forwarded to the wrapped VecEnv.

        Args:
            seed: The random seed. May be None for completely random seeding.

        Returns:
            A list containing the seeds for each individual env. Note that all list
            elements may be None, if the env does not return anything when seeded.
        """
        self.rng = np.random.default_rng(seed=seed)
        return list(self.venv.seed(seed))

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        self.traj_accum = rollout_multi_robot.TrajectoryAccumulator()
        obs = self.venv.reset()
        assert isinstance(obs, np.ndarray)
        for i, ob in enumerate(obs):
            self.traj_accum.add_step({"obs": ob}, key=i)
        self._last_obs = obs
        self._is_reset = True
        self._last_user_actions = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Steps with a `1 - beta` chance of using `self.get_robot_acts` instead.

        DAgger needs to be able to inject imitation policy actions randomly at some
        subset of time steps. This method has a `self.beta` chance of keeping the
        `actions` passed in as an argument, and a `1 - self.beta` chance of
        forwarding actions generated by `self.get_robot_acts` instead.
        "robot" (i.e. imitation policy) action if necessary.

        At the end of every episode, a `TrajectoryWithRew` is saved to `self.save_dir`,
        where every saved action is the expert action, regardless of whether the
        robot action was used during that timestep.

        Args:
            actions: the _intended_ demonstrator/expert actions for the current
                state. This will be executed with probability `self.beta`.
                Otherwise, a "robot" (typically a BC policy) action will be sampled
                and executed instead via `self.get_robot_act`.
        """
        assert self._is_reset, "call .reset() before .step()"
        assert self._last_obs is not None

        # Replace each given action with a robot action 100*(1-beta)% of the time.
        actual_acts = np.array(actions)

        mask = self.rng.uniform(0, 1, size=(self.num_envs,)) > self.beta
        if np.sum(mask) != 0:
            # todo: check if this makes sense for multiple venv
            # acts_conc = np.array()
            # for n in range(self.n_robots):
            # acts_nth_robot = self.get_robot_acts(self._last_obs[mask, n, :])
            # acts_conc = np.concatenate((acts_conc, acts_nth_robot), axis=1)
            #

            acts_list = [self.get_robot_acts(self._last_obs[mask, n, :]) for n in range(self.n_robots)]
            actual_acts[mask] = np.hstack(acts_list)

        self._last_user_actions = actions
        self.venv.step_async(actual_acts)

    def step_wait(self) -> VecEnvStepReturn:
        """Returns observation, reward, etc after previous `step_async()` call.

        Stores the transition, and saves trajectory as demo once complete.

        Returns:
            Observation, reward, dones (is terminal?) and info dict.
        """
        next_obs, rews, dones, infos = self.venv.step_wait()
        assert isinstance(next_obs, np.ndarray)
        assert self.traj_accum is not None
        assert self._last_user_actions is not None
        self._last_obs = next_obs
        fresh_demos = self.traj_accum.add_steps_and_auto_finish(
            obs=next_obs,
            acts=self._last_user_actions,
            rews=rews,
            infos=infos,
            dones=dones,
        )
        # todo: split reward
        for traj_index, traj in enumerate(fresh_demos):
            for n in range(self.n_robots):
                obs_nth_robot = traj.obs[:, n, :]
                acts_nth_robot = traj.acts[:,
                                 n * self.actions_size_single_robot:(n + 1) * self.actions_size_single_robot]
                # todo: reward and info per robot
                traj_nth_robot = types.TrajectoryWithRew(
                    rews=traj.rews, terminal=traj.terminal, obs=obs_nth_robot, acts=acts_nth_robot, infos=traj.infos
                )
                _save_dagger_demo(traj_nth_robot, traj_index * 10 + n + 1, self.save_dir, self.rng)

        return next_obs, rews, dones, infos


class ThriftyTrajectoryCollectorMultiRobot(vec_env.VecEnvWrapper):
    """DAgger VecEnvWrapper for querying and saving expert actions.

    Every call to `.step(actions)` accepts and saves expert actions to `self.save_dir`,
    but only forwards expert actions to the wrapped VecEnv with probability
    `self.beta`. With probability `1 - self.beta`, a "robot" action (i.e
    an action from the imitation policy) is forwarded instead.

    Demonstrations are saved as `TrajectoryWithRew` to `self.save_dir` at the end
    of every episode.
    """

    traj_accum: Optional[rollout_multi_robot.TrajectoryAccumulator]
    _last_obs: Optional[np.ndarray]
    _last_user_actions: Optional[np.ndarray]

    def __init__(
            self,
            venv: vec_env.VecEnv,
            get_robot_acts: Callable[[np.ndarray], np.ndarray],
            save_dir: types.AnyPath,
            rng: np.random.Generator,
            variance: Callable[[], float],
            switch2human_thresh: [np.ndarray],
            switch2human_thresh2: [np.ndarray],
            switch2robot_thresh: [np.ndarray],
            switch2robot_thresh2: [np.ndarray],
            n_robots,
            actions_size_single_robot,
            is_initial_collection: bool = False,

    ) -> None:
        """Builds InteractiveTrajectoryCollector.

        Args:
            venv: vectorized environment to sample trajectories from.
            get_robot_acts: get robot actions that can be substituted for
                human actions. Takes a vector of observations as input & returns a
                vector of actions.
            beta: fraction of the time to use action given to .step() instead of
                robot action. The choice of robot or human action is independently
                randomized for each individual `Env` at every timestep.
            save_dir: directory to save collected trajectories in.
            rng: random state for random number generation.
        """
        super().__init__(venv)
        self.get_robot_acts = get_robot_acts
        self.variance = variance
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None
        self.rng = rng
        self.is_initial_collection = is_initial_collection
        self.switch2human_thresh = switch2human_thresh
        self.switch2human_thresh2 = switch2human_thresh2
        self.switch2robot_thresh = switch2robot_thresh
        self.switch2robot_thresh2 = switch2robot_thresh2
        self.expert_mode = [[False] * n_robots] * self.venv.num_envs
        self.estimates = [[] * n_robots] * self.venv.num_envs
        self.estimates2 = [[] * n_robots] * self.venv.num_envs
        self.target_rate = 0.01
        self.q_learning = False
        self.num_switch_to_robot = 0
        self.num_switch_to_human = 0
        self.num_switch_to_human2 = 0
        self.online_burden = 0
        self.n_robots = n_robots
        self.actions_size_single_robot = actions_size_single_robot

        # for _ in self.venv.envs:

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set the seed for the DAgger random number generator and wrapped VecEnv.

        The DAgger RNG is used along with `self.beta` to determine whether the expert
        or robot action is forwarded to the wrapped VecEnv.

        Args:
            seed: The random seed. May be None for completely random seeding.

        Returns:
            A list containing the seeds for each individual env. Note that all list
            elements may be None, if the env does not return anything when seeded.
        """
        self.rng = np.random.default_rng(seed=seed)
        return list(self.venv.seed(seed))

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        self.traj_accum = rollout_multi_robot.TrajectoryAccumulator()
        obs = self.venv.reset()
        assert isinstance(obs, np.ndarray)
        for i, ob in enumerate(obs):
            self.traj_accum.add_step({"obs": ob}, key=i)
        self._last_obs = obs
        self._is_reset = True
        self._last_user_actions = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Steps with a `1 - beta` chance of using `self.get_robot_acts` instead.

        DAgger needs to be able to inject imitation policy actions randomly at some
        subset of time steps. This method has a `self.beta` chance of keeping the
        `actions` passed in as an argument, and a `1 - self.beta` chance of
        forwarding actions generated by `self.get_robot_acts` instead.
        "robot" (i.e. imitation policy) action if necessary.

        At the end of every episode, a `TrajectoryWithRew` is saved to `self.save_dir`,
        where every saved action is the expert action, regardless of whether the
        robot action was used during that timestep.

        Args:
            actions: the _intended_ demonstrator/expert actions for the current
                state. This will be executed with probability `self.beta`.
                Otherwise, a "robot" (typically a BC policy) action will be sampled
                and executed instead via `self.get_robot_act`.
        """
        assert self._is_reset, "call .reset() before .step()"
        assert self._last_obs is not None

        # Replace each given action with a robot action 100*(1-beta)% of the time.
        actual_acts = np.array(actions)
        # TODO: get variance per env predict
        # TODO: q learning if ensemble

        if not self.is_initial_collection:
            for env_idx, env_act in enumerate(actual_acts):
                for n in range(self.n_robots):
                    print(f'env: {env_idx}, robot: {n}, expert mode: {self.expert_mode[env_idx][n]}')
                    robot_act = self.get_robot_acts(self._last_obs[env_idx, n])
                    safety = 0
                    # safety = ac.safety(o,a)
                    variance = self.variance()
                    if not self.expert_mode[env_idx][n]:
                        self.estimates[env_idx][n].append(self.variance())
                        # TODO : add safety
                        # self.estimates2[i].append(policy.safety())
                        self.estimates2[env_idx][n].append(0)
                        actual_acts[env_idx, n * self.actions_size_single_robot:(n + 1) * self.actions_size_single_robot] = robot_act
                    if self.expert_mode[env_idx][n]:
                        self.online_burden += 1

                        # self.risk[i].append(safety)
                        # safety = policy.safety
                        if (np.sum(
                                (robot_act - env_act[n * self.actions_size_single_robot:(n + 1) * self.actions_size_single_robot]) ** 2
                        ) < self.switch2robot_thresh[env_idx] and
                                (not self.q_learning or safety > self.switch2robot_thresh2[env_idx][n])):
                            print("Switch to Robot")
                            self.expert_mode[env_idx][n] = False
                            self.num_switch_to_robot += 1
                    # TODO: change to > whith proper variance
                    elif self.variance() >= self.switch2human_thresh[env_idx][n]:
                        print("Switch to Human (Novel)")
                        self.num_switch_to_human += 1
                        self.expert_mode[env_idx][n] = True
                        continue
                    elif self.q_learning and safety < self.switch2human_thresh2[env_idx][n]:
                        print("Switch to Human (Risk)")
                        self.num_switch_to_human2 += 1
                        self.expert_mode[env_idx][n] = True
                        continue
                    else:
                        # self.risk[i].append(safety)
                        print("")

        self._last_user_actions = actions
        self.venv.step_async(actual_acts)

    def step_wait(self) -> VecEnvStepReturn:
        """Returns observation, reward, etc after previous `step_async()` call.

        Stores the transition, and saves trajectory as demo once complete.

        Returns:
            Observation, reward, dones (is terminal?) and info dict.
        """
        next_obs, rews, dones, infos = self.venv.step_wait()
        assert isinstance(next_obs, np.ndarray)
        assert self.traj_accum is not None
        assert self._last_user_actions is not None
        self._last_obs = next_obs
        fresh_demos = self.traj_accum.add_steps_and_auto_finish(
            obs=next_obs,
            acts=self._last_user_actions,
            rews=rews,
            infos=infos,
            dones=dones,
        )
        # todo: split reward
        for traj_index, traj in enumerate(fresh_demos):
            for n in range(self.n_robots):
                obs_nth_robot = traj.obs[:, n, :]
                acts_nth_robot = traj.acts[:,
                                 n * self.actions_size_single_robot:(n + 1) * self.actions_size_single_robot]
                # todo: reward and info per robot
                traj_nth_robot = types.TrajectoryWithRew(
                    rews=traj.rews, terminal=traj.terminal, obs=obs_nth_robot, acts=acts_nth_robot, infos=traj.infos
                )
                _save_dagger_demo(traj_nth_robot, traj_index * 10 + n + 1, self.save_dir, self.rng)

        return next_obs, rews, dones, infos


    def recompute_thresholds(self):
        switch2human_thresh = [[] * self.venv.num_envs]
        switch2human_thresh2 = [[] * self.venv.num_envs]
        switch2robot_thresh2 = [[] * self.venv.num_envs]
        for env_idx in range(self.venv.num_envs):
            for n in range(self.n_robots):
                if len(self.estimates[env_idx][n]) > 25:
                    target_idx = int((1 - self.target_rate) * len(self.estimates[env_idx][n]))
                    switch2human_thresh[env_idx].append(sorted(self.estimates[env_idx][n])[target_idx])
                    switch2human_thresh2[env_idx].append(sorted(self.estimates2[env_idx][n], reverse=True)[target_idx])
                    switch2robot_thresh2[env_idx].append(sorted(self.estimates2[env_idx][n])[int(0.5 * len(self.estimates[env_idx][n]))])

                 print("len(estimates): {}, New switch thresholds: {} {} {}".format(len(self.estimates[env_idx]),
                                                                                    switch2human_thresh[env_idx],
                                                                                    switch2human_thresh2[env_idx],
                                                                                    switch2robot_thresh2)[env_idx])

        return switch2human_thresh, switch2human_thresh2, switch2robot_thresh2


    def estimate_switch_parameters(
            self,
            data,
            target_rate=0.01
    ):
        """estimate switch-back parameter and initial switch-to parameter from data

        Returns:
            None.
        """
        discrepancies, estimates = [], []
        heldout_thresh = int(0.9 * data.acts.shape[0])
        for i in range(0, heldout_thresh):
            a_pred = self.get_robot_acts(data.obs[i])
            a_sup = data.acts[i]
            discrepancies.append(np.sum((a_pred - a_sup) ** 2))
            # TODO append variance for ensemble (policy.variance(obs))
            estimates.append(self.variance())
        heldout_discrepancies, heldout_estimates = [], []
        for i in range(heldout_thresh, data.acts.shape[0]):
            a_pred = self.get_robot_acts(data.obs[i])
            a_sup = data.acts[i]
            heldout_discrepancies.append(np.sum((a_pred - a_sup) ** 2))
            # TODO append variance for ensemble (policy.variance(obs))
            heldout_estimates.append(self.variance())
        switch2robot_thresh = [np.array(discrepancies).mean()] * self.venv.num_envs
        target_idx = int((1 - target_rate) * len(heldout_estimates))
        switch2human_thresh = [sorted(heldout_estimates)[target_idx]] * self.venv.num_envs
        print("Estimated switch-back threshold: {}".format(self.switch2robot_thresh))
        print("Estimated switch-to threshold: {}".format(self.switch2human_thresh))
        switch2human_thresh2 = [
                                   0.48] * self.venv.num_envs  # a priori guess: 48% discounted probability of success. Could also estimate from data
        switch2robot_thresh2 = [0.495] * self.venv.num_envs

        return switch2robot_thresh, switch2human_thresh, switch2human_thresh2, switch2robot_thresh2


class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""


class DAggerTrainerMultiRobot(base.BaseImitationAlgorithm):
    """DAgger training class with low-level API suitable for interactive human feedback.

    In essence, this is just BC with some helpers for incrementally
    resuming training and interpolating between demonstrator/learnt policies.
    Interaction proceeds in "rounds" in which the demonstrator first provides a
    fresh set of demonstrations, and then an underlying `BC` is invoked to
    fine-tune the policy on the entire set of demonstrations collected in all
    rounds so far. Demonstrations and policy/trainer checkpoints are stored in a
    directory with the following structure::

       scratch-dir-name/
           checkpoint-001.pt
           checkpoint-002.pt
           …
           checkpoint-XYZ.pt
           checkpoint-latest.pt
           demos/
               round-000/
                   demos_round_000_000.npz
                   demos_round_000_001.npz
                   …
               round-001/
                   demos_round_001_000.npz
                   …
               …
               round-XYZ/
                   …
    """

    _all_demos: List[types.Trajectory]

    DEFAULT_N_EPOCHS: int = 4
    """The default number of BC training epochs in `extend_and_update`."""

    def __init__(
            self,
            *,
            venv: vec_env.VecEnv,
            scratch_dir: types.AnyPath,
            rng: np.random.Generator,
            beta_schedule: Optional[Callable[[int], float]] = None,
            bc_trainer: bc_multi_robot.BCMultiRobot,
            custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
            n_robots: int
    ):
        """Builds DAggerTrainer.

        Args:
            venv: Vectorized training environment.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            rng: random state for random number generation.
            beta_schedule: Provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
            bc_trainer: A `BC` instance used to train the underlying policy.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)
        self.beta_schedule = beta_schedule
        self.scratch_dir = util.parse_path(scratch_dir)
        self.venv = venv
        self.round_num = 0
        self._last_loaded_round = -1
        self._all_demos = []
        self.rng = rng

        # TODO check check
        observation_space_shape = (n_robots, bc_trainer.observation_space.shape[0])
        # print("observation_space_shape: {}".format(observation_space_shape))
        # check_observation_space = Space(shape=observation_space_shape, dtype=bc_trainer.observation_space.dtype)

        action_space_shape = (n_robots * bc_trainer.action_space.shape[0],)
        # print("action_space_shape: {}".format(action_space_shape))
        # check_action_space = Space(shape=action_space_shape, dtype=bc_trainer.action_space.dtype)

        check_observation_space = Box(low=-np.inf, high=np.inf,
                                      shape=(n_robots, bc_trainer.observation_space.shape[0]), dtype=np.float64)
        check_action_space = Box(low=-10.0, high=10.0, shape=(n_robots * bc_trainer.action_space.shape[0],),
                                 dtype=np.float64)

        utils.check_for_correct_spaces(
            self.venv,
            check_observation_space,
            check_action_space,
        )
        self.bc_trainer = bc_trainer
        self.bc_trainer.logger = self.logger

    def __getstate__(self):
        """Return state excluding non-pickleable objects."""
        d = dict(self.__dict__)
        del d["venv"]
        del d["_logger"]
        return d

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        """Returns logger for this object."""
        return super().logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        # DAgger and inner-BC logger should stay in sync
        self._logger = value
        self.bc_trainer.logger = value

    @property
    def policy(self) -> policies.BasePolicy:
        return self.bc_trainer.policy

    @property
    def batch_size(self) -> int:
        return self.bc_trainer.batch_size

    def _load_all_demos(self) -> Tuple[types.Transitions, List[int]]:
        num_demos_by_round = []
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)
            self._all_demos.extend(serialize.load(p)[0] for p in demo_paths)
            num_demos_by_round.append(len(demo_paths))
        logging.info(f"Loaded {len(self._all_demos)} total")
        demo_transitions = rollout_multi_robot.flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round

    def _get_demo_paths(self, round_dir: pathlib.Path) -> List[pathlib.Path]:
        # listdir returns filenames in an arbitrary order that depends on the
        # file system implementation:
        # https://stackoverflow.com/questions/31534583/is-os-listdir-deterministic
        # To ensure the order is consistent across file systems,
        # we sort by the filename.
        filenames = sorted(os.listdir(round_dir))
        return [round_dir / f for f in filenames if f.endswith(".npz")]

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> pathlib.Path:
        if round_num is None:
            round_num = self.round_num
        return self.scratch_dir / "demos" / f"round-{round_num:03d}"

    def _try_load_demos(self) -> None:
        """Load the dataset for this round into self.bc_trainer as a DataLoader."""
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if demo_dir.is_dir() else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".create_trajectory_collector()",
            )

        if self._last_loaded_round < self.round_num:
            transitions, num_demos = self._load_all_demos()
            logging.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds",
            )
            if len(transitions) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(transitions)}",
                )
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            self.bc_trainer.set_demonstrations(data_loader)
            self._last_loaded_round = self.round_num

    def extend_and_update(
            self,
            bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        if "log_rollouts_venv" not in user_keys:
            bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        logging.info("Loading demonstrations")
        self._try_load_demos()
        logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        logging.info(f"New round number is {self.round_num}")
        return self.round_num

    def create_trajectory_collector_multi_robot(self,
                                                actions_size_single_robot: int,
                                                n_robots: int = 1,
                                                ) -> InteractiveTrajectoryCollectorMultiRobot:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()
        beta = self.beta_schedule(self.round_num)
        collector = InteractiveTrajectoryCollectorMultiRobot(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            beta=beta,
            save_dir=save_dir,
            rng=self.rng,
            n_robots=n_robots,
            actions_size_single_robot=actions_size_single_robot,

        )
        return collector

    def create_thrifty_trajectory_collector_multi_robot(
            self, switch2robot_thresh,
            switch2human_thresh,
            switch2human_thresh2,
            switch2robot_thresh2,
            actions_size_single_robot: int,
            n_robots: int = 1,
            is_initial_collection=False,
    ) -> ThriftyTrajectoryCollectorMultiRobot:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()

        collector = ThriftyTrajectoryCollectorMultiRobot(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            save_dir=save_dir,
            rng=self.rng,
            variance=lambda: self.bc_trainer.get_policy_var(),
            switch2robot_thresh=switch2robot_thresh,
            switch2human_thresh=switch2human_thresh,
            switch2human_thresh2=switch2human_thresh2,
            switch2robot_thresh2=switch2robot_thresh2,
            n_robots=n_robots,
            actions_size_single_robot=actions_size_single_robot,
            is_initial_collection=is_initial_collection,

        )
        return collector

    def create_thrifty_trajectory_collector(
            self, switch2robot_thresh,
            switch2human_thresh,
            switch2human_thresh2,
            switch2robot_thresh2,
            is_initial_collection=False
    ) -> ThriftyTrajectoryCollectorMultiRobot:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()

        collector = ThriftyTrajectoryCollectorMultiRobot(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            save_dir=save_dir,
            rng=self.rng,
            variance=lambda: self.bc_trainer.get_policy_var(),
            switch2robot_thresh=switch2robot_thresh, switch2human_thresh=switch2human_thresh,
            switch2human_thresh2=switch2human_thresh2, switch2robot_thresh2=switch2robot_thresh2,
            is_initial_collection=is_initial_collection
        )
        return collector

    def save_trainer(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """Create a snapshot of trainer in the scratch/working directory.

        The created snapshot can be reloaded with `reconstruct_trainer()`.
        In addition to saving one copy of the policy in the trainer snapshot, this
        method saves a second copy of the policy in its own file. Having a second copy
        of the policy is convenient because it can be loaded on its own and passed to
        evaluation routines for other algorithms.

        Returns:
            checkpoint_path: a path to one of the created `DAggerTrainer` checkpoints.
            policy_path: a path to one of the created `DAggerTrainer` policies.
        """
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        # save full trainer checkpoints
        checkpoint_paths = [
            self.scratch_dir / f"checkpoint-{self.round_num:03d}.pt",
            self.scratch_dir / "checkpoint-latest.pt",
        ]
        for checkpoint_path in checkpoint_paths:
            th.save(self, checkpoint_path)

        # save policies separately for convenience
        policy_paths = [
            self.scratch_dir / f"policy-{self.round_num:03d}.pt",
            self.scratch_dir / "policy-latest.pt",
        ]
        for policy_path in policy_paths:
            util.save_policy(self.policy, policy_path)

        return checkpoint_paths[0], policy_paths[0]
