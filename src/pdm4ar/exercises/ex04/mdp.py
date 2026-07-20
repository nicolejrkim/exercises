from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Cell, Policy, State, ValueFunc

# reward/cost constants (unit: 1k USD)
GOAL_REWARD = 50.0
"""Bonus received for every stage the robot surveys the goal."""
DEPLOY_COST = 10.0
"""Cost of deploying a new robot from the START cell."""
WONDERLAND_REWARD = 3.0
"""Bonus for passing through a WONDERLAND, only if the teleport does not end in a CLIFF/outside."""

MOVES: dict[Action, tuple[int, int]] = {
    Action.NORTH: (-1, 0),
    Action.WEST: (0, -1),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
}
"""Displacement (di, dj) associated to each movement action."""

CELL_TIME: dict[Cell, float] = {
    Cell.GOAL: 1.0,
    Cell.START: 1.0,
    Cell.GRASS: 1.0,
    Cell.SWAMP: 2.0,
    Cell.WONDERLAND: 1.0,
    Cell.CLIFF: 1.0,
}
"""Hours needed to (try to) leave a cell of the given type."""


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    # ---------------------------------------------------------------- helpers

    @cached_property
    def start(self) -> State:
        idx = np.argwhere(self.grid == Cell.START)
        assert len(idx) == 1, "The map must contain exactly one START cell"
        return int(idx[0][0]), int(idx[0][1])

    def _inside(self, state: State) -> bool:
        return 0 <= state[0] < self.grid.shape[0] and 0 <= state[1] < self.grid.shape[1]

    def _is_fatal(self, state: State) -> bool:
        """A cell outside of the map or a CLIFF breaks the robot down."""
        return not self._inside(state) or self.grid[state] == Cell.CLIFF

    def get_admissible_actions(self, state: State) -> list[Action]:
        """Actions the robot is allowed to pick in the given state."""
        if self.grid[state] == Cell.GOAL:
            return [Action.STAY]
        actions = [a for a, d in MOVES.items() if not self._is_fatal((state[0] + d[0], state[1] + d[1]))]
        actions.append(Action.ABANDON)
        return actions

    def _outcome_distribution(self, state: State, action: Action) -> dict[Optional[Action], float]:
        """
        Probability of each "physical outcome" of the chosen action, before resolving where the robot lands.
        A movement action key means the robot moved in that direction, `None` means it broke down on the spot,
        and a missing key with the leftover probability means it did not manage to leave the cell.
        """
        cell = self.grid[state]
        if cell == Cell.SWAMP:
            dist: "dict[Optional[Action], float]" = {a: (0.5 if a == action else 0.25 / 3) for a in MOVES}
            dist[None] = 0.05
        else:
            dist = {a: (0.75 if a == action else 0.25 / 3) for a in MOVES}
        return dist

    def _stay_prob(self, state: State) -> float:
        """Probability of not being able to leave the cell when a movement action is chosen."""
        return 0.2 if self.grid[state] == Cell.SWAMP else 0.0

    def _transitions(self, state: State, action: Action) -> dict[State, tuple[float, float]]:
        """
        Full outcome of taking `action` in `state`.

        :return: {next_state: (probability, expected reward contribution)}, where the reward contribution is
                 already weighted by the probability of the specific transition.
        """
        out: dict[State, tuple[float, float]] = {}

        def add(next_state: State, prob: float, reward: float) -> None:
            p, r = out.get(next_state, (0.0, 0.0))
            out[next_state] = (p + prob, r + prob * reward)

        if action not in self.get_admissible_actions(state):
            return out

        if action == Action.STAY:
            add(state, 1.0, GOAL_REWARD)
            return out
        if action == Action.ABANDON:
            add(self.start, 1.0, -DEPLOY_COST)
            return out

        time_cost = CELL_TIME[Cell(self.grid[state])]
        for outcome, prob in self._outcome_distribution(state, action).items():
            if prob == 0.0:
                continue
            if outcome is None:  # broke down while trying to move
                add(self.start, prob, -time_cost - DEPLOY_COST)
                continue
            d = MOVES[outcome]
            target = (state[0] + d[0], state[1] + d[1])
            if self._is_fatal(target):
                add(self.start, prob, -time_cost - DEPLOY_COST)
            elif self.grid[target] == Cell.WONDERLAND:
                # instantaneous teleport to an adjacent cell, excluding the one the robot came from
                landings = [(target[0] + dd[0], target[1] + dd[1]) for dd in MOVES.values()]
                landings = [land for land in landings if land != state]
                for land in landings:
                    p = prob / len(landings)
                    if self._is_fatal(land):
                        add(self.start, p, -time_cost - DEPLOY_COST)
                    else:
                        add(land, p, -time_cost + WONDERLAND_REWARD)
            else:
                add(target, prob, -time_cost)

        stay_prob = self._stay_prob(state)
        if stay_prob > 0.0:
            add(state, stay_prob, -time_cost)
        return out

    # ------------------------------------------------------------- interfaces

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        return self._transitions(state, action).get(next_state, (0.0, 0.0))[0]

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        prob, weighted_reward = self._transitions(state, action).get(next_state, (0.0, 0.0))
        return weighted_reward / prob if prob > 0.0 else 0.0

    # ------------------------------------------------- compiled model (solvers)

    @cached_property
    def model(self) -> "MdpModel":
        """Flattened transition model, shared by both solvers."""
        return MdpModel(self)


class MdpModel:
    """
    Dense-but-padded representation of the MDP, tailored for vectorised dynamic programming.

    States are all the cells the robot can actually decide in, i.e. every cell except CLIFF (unreachable as a
    decision point) and WONDERLAND (crossed instantaneously, no action is ever taken there).
    """

    def __init__(self, mdp: GridMdp):
        grid = mdp.grid
        self.gamma = mdp.gamma
        self.shape = grid.shape

        decision_mask = (grid != Cell.CLIFF) & (grid != Cell.WONDERLAND)
        self.states: list[State] = [(int(i), int(j)) for i, j in np.argwhere(decision_mask)]
        self.state_index = np.full(grid.shape, -1, dtype=np.int64)
        for k, s in enumerate(self.states):
            self.state_index[s] = k
        n_states = len(self.states)

        self.actions: list[Action] = list(Action)
        n_actions = len(self.actions)

        transitions = [[mdp._transitions(s, a) for a in self.actions] for s in self.states]
        self.admissible = np.array([[len(t) > 0 for t in row] for row in transitions], dtype=bool)
        width = max(1, max(len(t) for row in transitions for t in row))

        # padded successor table: entries beyond the actual support have probability 0 and point at state 0
        self.next_idx = np.zeros((n_states, n_actions, width), dtype=np.int64)
        self.probs = np.zeros((n_states, n_actions, width), dtype=np.float64)
        self.reward = np.zeros((n_states, n_actions), dtype=np.float64)
        for k, row in enumerate(transitions):
            for a, trans in enumerate(row):
                for slot, (next_state, (prob, weighted_reward)) in enumerate(trans.items()):
                    self.next_idx[k, a, slot] = self.state_index[next_state]
                    self.probs[k, a, slot] = prob
                    self.reward[k, a] += weighted_reward

        # Q-values of inadmissible actions must never win the maximisation
        self.mask = np.where(self.admissible, 0.0, -np.inf)

    def q_values(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        """Q(s, a) = R(s, a) + gamma * sum_s' P(s'|s, a) V(s') for every state-action pair."""
        expected_next = np.einsum("sak,sak->sa", self.probs, value[self.next_idx])
        return self.reward + self.gamma * expected_next + self.mask

    def policy_backup(self, value: NDArray[np.float64], policy: NDArray[np.int64]) -> NDArray[np.float64]:
        """One evaluation sweep of the given (deterministic) policy."""
        rows = np.arange(len(self.states))
        idx = self.next_idx[rows, policy]
        prob = self.probs[rows, policy]
        return self.reward[rows, policy] + self.gamma * np.einsum("sk,sk->s", prob, value[idx])

    def greedy(self, value: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.argmax(self.q_values(value), axis=1).astype(np.int64)

    def to_grids(self, value: NDArray[np.float64], policy: NDArray[np.int64]) -> tuple[ValueFunc, Policy]:
        """Scatter the flat vectors back onto the grid (CLIFF/WONDERLAND cells stay at their default)."""
        value_func = np.zeros(self.shape, dtype=np.float64)
        policy_grid = np.zeros(self.shape, dtype=np.int64)
        rows = np.array([s[0] for s in self.states], dtype=np.int64)
        cols = np.array([s[1] for s in self.states], dtype=np.int64)
        value_func[rows, cols] = value
        policy_grid[rows, cols] = policy
        return value_func, policy_grid


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        pass
