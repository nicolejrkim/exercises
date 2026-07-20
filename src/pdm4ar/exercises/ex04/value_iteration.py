import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function

TOLERANCE = 1e-10
MAX_ITERATIONS = 10_000


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        model = grid_mdp.model
        value = np.zeros(len(model.states), dtype=np.float64)

        for _ in range(MAX_ITERATIONS):
            updated = np.max(model.q_values(value), axis=1)
            delta = np.max(np.abs(updated - value))
            value = updated
            if delta < TOLERANCE:
                break

        return model.to_grids(value, model.greedy(value))
