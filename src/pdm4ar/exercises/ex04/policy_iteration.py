import numpy as np
from numpy.typing import NDArray

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver, MdpModel
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function

try:  # exact policy evaluation via a sparse linear solve, iterative fallback if scipy is missing
    from scipy.sparse import csr_matrix, identity
    from scipy.sparse.linalg import spsolve

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

TOLERANCE = 1e-10
MAX_ITERATIONS = 1_000
MAX_EVAL_SWEEPS = 10_000


def _evaluate_policy(model: MdpModel, policy: NDArray[np.int64], value: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Solve V = R_pi + gamma * P_pi @ V exactly.

    (I - gamma * P_pi) is strictly diagonally dominant for gamma < 1, hence invertible. We assemble it as a
    sparse matrix and factorise it; if scipy is unavailable we fall back to iterative evaluation.
    """
    if not HAS_SCIPY:
        return _evaluate_policy_iteratively(model, policy, value)

    n = len(model.states)
    rows = np.arange(n)
    cols = model.next_idx[rows, policy]
    data = model.probs[rows, policy]
    row_idx = np.repeat(rows, cols.shape[1])
    p_pi = csr_matrix((data.ravel(), (row_idx, cols.ravel())), shape=(n, n))
    return spsolve((identity(n, format="csr") - model.gamma * p_pi).tocsc(), model.reward[rows, policy])


def _evaluate_policy_iteratively(
    model: MdpModel, policy: NDArray[np.int64], value: NDArray[np.float64]
) -> NDArray[np.float64]:
    for _ in range(MAX_EVAL_SWEEPS):
        updated = model.policy_backup(value, policy)
        delta = np.max(np.abs(updated - value))
        value = updated
        if delta < TOLERANCE:
            break
    return value


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        model = grid_mdp.model
        value = np.zeros(len(model.states), dtype=np.float64)
        # start from an arbitrary admissible policy
        policy = np.argmax(model.admissible, axis=1).astype(np.int64)

        for _ in range(MAX_ITERATIONS):
            value = _evaluate_policy(model, policy, value)
            improved = model.greedy(value)
            if np.array_equal(improved, policy):
                break
            policy = improved

        return model.to_grids(value, policy)
