from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3 import EpsilonGreedyPolicy


class SARSALambdaAgent(AbstractAgent):
    """SARSA(\lambda) algorithm"""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        lam: float = 0.5,
    ) -> None:
        """Initialize the SARSA agent.

        Parameters
        ----------
        env : gym.Env
            The environment in which the agent will interact.
        policy : EpsilonGreedyPolicy
            Policy for selecting actions, typically epsilon-greedy.
        alpha : float
            Learning rate (step size for updates), by default 0.5.
        gamma : float
            Discount factor for future rewards, by default 1.0.
        lam : float
            Weight factor between TD(0) target and MC target

        Raises
        ------
        AssertionError
            If `gamma` is not in [0, 1] or if `alpha` is not positive.
        """

        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert 0 <= lam <= 1, "Lambda should be in [0, 1]"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lam = lam

        # number of actions → used by Q’s default factory
        self.n_actions = env.action_space.n

        # Build Q so that unseen states map to zero‐vectors
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        # Build an eligibility trace to track last seen state-action pairs
        self.trace: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        self.policy = policy

    def predict_action(self, state: np.array, evaluate: bool = False) -> Any:  # type: ignore # noqa
        """Select an action for the given state using the policy.

        Parameters
        ----------
        state : np.array
            The current state.
        evaluate : bool, optional
            If True, use the greedy policy without exploration; otherwise use exploration.

        Returns
        -------
        Any
            The selected action.
        """
        return self.policy(self.Q, state, evaluate=evaluate)

    def save(self, path: str) -> Any:  # type: ignore
        """Save the learned Q-table to a file.

        Parameters
        ----------
        path : str
            Path to the file where the Q-table will be saved (.npy format).
        """
        np.save(path, self.Q)  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table

        Parameters
        ----------
        path :
            Path to saved the Q table

        """
        self.Q = np.load(path)

    def update_agent(  # type: ignore
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool,
    ) -> float:
        """Update the Q-value for a state-action pair using the SARSA(\lambda) update rule.

        Parameters
        ----------
        state : State
            The current state.
        action : int
            The action taken in the current state.
        reward : float
            The reward received after taking the action.
        next_state : State
            The next state after taking the action.
        next_action : int
            The action selected in the next state.
        done : bool
            True if the episode has ended; False otherwise.
        """

        # Calculate classical TD(0) target
        delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]

        # Manage eligibility trace, essentially tracking which state-action-pairs recently occured
        # (as these values are later decreased)
        # Choose between 3 modes accumulate, dutch, and replace
        type = "accumulate"
        if type == "accumulate":
            self.trace[state][action] += 1
        elif type == "dutch":
            self.trace[state][action] = (1 - self.alpha) * self.trace[state][action] + 1
        elif type == "replace":
            self.trace[state][action] = 1
        else:
            assert False

        for state in self.Q.keys():
            for action in range(self.n_actions):
                # Apply TD update but take eligibility trace into account!
                self.Q[state][action] = self.Q[state][action] + self.alpha * delta * self.trace[state][action]
                # Gradually decrease eligibility traces
                self.trace[state][action] = self.gamma * self.lam * self.trace[state][action]
