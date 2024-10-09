import numpy as np
from amplify import (
    VariableGenerator,
    FixstarsClient,
    Model,
    PolyArray,
    solve,
    sum,
    less_equal,
)
from util import CQKPInstance, CQKPSolution
from typing import Optional
from API_KEY import AMPLIFY_TOKEN


def evaluate_objective(instance: CQKPInstance, X: np.ndarray) -> float:
    N = instance.N
    Q = instance.Q
    L = instance.L
    Indices = [(i, j) for i in range(N) for j in range(N) if Q[i, j] > 0]
    obj_lin = sum(L[i] * X[i] for i in range(N) if L[i] > 0)
    obj_qua = sum(Q[i, j] * X[i] * X[j] for (i, j) in Indices)
    return obj_lin + obj_qua


def naive_formulation(instance: CQKPInstance) -> tuple[Model, PolyArray]:
    N = instance.N
    L = instance.L
    A = instance.A
    Q = instance.Q
    B = instance.b
    K = instance.k

    # Indices = (i, j) \in [N] x [N] s.t. i < j
    Indices = [(i, j) for i in range(N) for j in range(N) if Q[i, j] > 0]

    # Variables X = {x[0],...,x[N-1]}
    gen = VariableGenerator()
    X = gen.array("Binary", N, name="x")

    obj_lin = sum(L[i] * X[i] for i in range(N) if L[i] > 0)
    obj_qua = sum(Q[i, j] * X[i] * X[j] for (i, j) in Indices)
    cons_weight = less_equal(sum(A[i] * X[i] for i in range(N)), B)
    cons_card = less_equal(sum(X), K)

    # Hamiltonian
    f = -obj_lin - obj_qua + cons_weight + cons_card
    return f, X


def run_single_experiment(instance: CQKPInstance) -> Optional[CQKPSolution]:
    f, X = naive_formulation(instance=instance)
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000  # 実行時間を 1000 ミリ秒に設定
    result = solve(f, client)

    # decode
    XX = X.evaluate(result.best.values)
    N = instance.N
    obj_value = evaluate_objective(instance, XX)
    Items = [i for i in range(N) if XX[i] > 0.5]

    result = CQKPSolution(Items, obj_value)
    return result


def run_multiple_experiments(
    instance: CQKPInstance, num_solves: int = 5
) -> list[Optional[CQKPSolution]]:
    N = instance.N
    f, X = naive_formulation(instance=instance)
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000  # 実行時間を 1000 ミリ秒に設定
    result = solve(f, client, num_solves=num_solves)


    res = np.zeros(result.num_solves)
    solutions = []
    for i in range(result.num_solves):
        XX = X.evaluate(result[i].values)
        obj_value = evaluate_objective(instance, XX)
        Items = [i for i in range(N) if XX[i] > 0.5]
        res[i] = obj_value
        solutions.append(
            CQKPSolution(Items, obj_value)
        )

    print(round(res.mean(), 3), round(res.std(), 3))
    return solutions