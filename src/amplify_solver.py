import numpy as np
from amplify import (
    VariableGenerator,
    FixstarsClient,
    AcceptableDegrees,
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
    A = instance.A
    Indices = [(i, j) for i in range(N) for j in range(N) if Q[i, j] > 0]
    obj_lin = sum(L[i] * X[i] for i in range(N) if L[i] > 0)
    obj_qua = sum(Q[i, j] * X[i] * X[j] for (i, j) in Indices)
    flag_card = sum(X) <= instance.k
    flag_cap = sum(A[i] * X[i] for i in range(N)) <= instance.b
    return {"obj": obj_lin + obj_qua, "card": flag_card, "cap": flag_cap}


def naive_formulation(
    instance: CQKPInstance,
    lambda_card: float = 1.0,
    lambda_cap: float = 1.0,
) -> tuple[Model, PolyArray]:
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
    f = -obj_lin - obj_qua + lambda_cap * cons_weight + lambda_card * cons_card
    return f, X


def formulation_linear(
    instance: CQKPInstance,
    lambda_card: float = 1.0,
    lambda_cap: float = 1.0,
) -> tuple[Model, PolyArray]:
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
    cons_weight = sum(A[i] * X[i] for i in range(N)) - B
    cons_card = sum(X) - K

    # Hamiltonian
    f = -obj_lin - obj_qua + lambda_cap * cons_weight + lambda_card * cons_card
    return f, X


def formulation_qubo_card(
    instance: CQKPInstance,
    lambda_card: float = 1.0,
    lambda_cap: float = 1.0,
) -> tuple[Model, PolyArray]:
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
    cons_weight = sum(A[i] * X[i] for i in range(N)) - B
    cons_card = (sum(X) - K) ** 2

    # Hamiltonian
    f = -obj_lin - obj_qua + lambda_cap * cons_weight + lambda_card * cons_card
    return f, X



def formulation_binary(
    instance: CQKPInstance,
    lambda_card: float = 1.0,
    lambda_cap: float = 1.0,
) -> tuple[Model, PolyArray]:
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
    S = gen.scalar("Integer", bounds=(0, B), name="s")

    obj_lin = sum(L[i] * X[i] for i in range(N) if L[i] > 0)
    obj_qua = sum(Q[i, j] * X[i] * X[j] for (i, j) in Indices)
    cons_card = (sum(X) - K) ** 2
    cons_weight = (sum(A[i] * X[i] for i in range(N)) + S - B) ** 2

    # Hamiltonian
    f = -obj_lin - obj_qua + lambda_cap * cons_weight + lambda_card * cons_card
    model = Model(f)
    bq = AcceptableDegrees(objective={"Binary": "Quadratic"})
    im, mapping = model.to_intermediate_model(bq, integer_encoding_method="Binary")
    return im, X, S


def formulation_unary(
    instance: CQKPInstance,
    lambda_card: float = 1.0,
    lambda_cap: float = 1.0,
    using_amplify: bool = True
) -> tuple[Model, PolyArray]:
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
    if not using_amplify:
        S = gen.array("Binary", B + 1, name="s")
    else:
        S = gen.scalar("Integer", bounds=(0, B), name="s")

    obj_lin = sum(L[i] * X[i] for i in range(N) if L[i] > 0)
    obj_qua = sum(Q[i, j] * X[i] * X[j] for (i, j) in Indices)
    cons_card = (sum(X) - K) ** 2
    if not using_amplify:
        # based on H^{qubo,unary}_{(CAP)}
        cons_weight = (sum(A[i] * X[i] for i in range(N)) + sum(S[i] for i in range(B + 1)) - B) ** 2
    else:
        # based on an integer variable supported by Fixstars Amplify
        cons_weight = (sum(A[i] * X[i] for i in range(N)) + S - B) ** 2

    # Hamiltonian
    f = -obj_lin - obj_qua + lambda_cap * cons_weight + lambda_card * cons_card
    return f, X, S


def run_single_experiment(instance: CQKPInstance) -> Optional[CQKPSolution]:
    f, X = naive_formulation(instance=instance)
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000  # 実行時間を 1000 ミリ秒に設定
    result = solve(f, client)

    # decode
    XX = X.evaluate(result.best.values)
    N = instance.N
    res = evaluate_objective(instance, XX)
    Items = [i for i in range(N) if XX[i] > 0.5]

    result = CQKPSolution(Items, res["obj"], res["card"], res["cap"])
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
        res = evaluate_objective(instance, XX)
        obj_value = res["obj"]
        Items = [i for i in range(N) if XX[i] > 0.5]
        res[i] = obj_value
        solutions.append(CQKPSolution(Items, obj_value, res["card"], res["cap"]))

    print(round(res.mean(), 3), round(res.std(), 3))
    return solutions
