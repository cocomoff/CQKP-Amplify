import os
import json
import numpy as np
import dataclasses
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class CQKPInstance:
    N: int  # number of items
    k: int  # cardinality (= selected items)
    b: float  # budget
    A: np.ndarray  # Numpy array of weights
    L: np.ndarray  # Numpy array of prices
    Q: np.ndarray  # Numpy matrix of quadratic prices


@dataclasses.dataclass
class CQKPSolution:
    Items: list # Selected items
    Obj: float # Objective value
    card: bool # Flag of cardinality constraint
    cap: bool # Flag of capacity constraint
    Slacks: tuple = () # Slack variables


def read_instance(filepath: Path) -> Optional[CQKPInstance]:
    if not filepath.exists:
        return None
    
    data = json.load(open(filepath, "r"))
    N = data["N"]
    K = data["k"]
    cb = data["b"]
    cA = data["A"]
    cl = data["l"]
    cQ = data["Q"]

    # build numpy array
    A = np.zeros(N)
    L = np.zeros(N)
    Q = np.zeros((N, N))

    for (i, j) in cA:
        A[i] = j
    for (i, j) in cl:
        L[i] = j
    for (i, j, k) in cQ:
        Q[i, j] = k
    return CQKPInstance(N, K, cb, A, L, Q)
