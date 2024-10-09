from amplify import VariableGenerator, FixstarsClient, solve, sum, less_equal
from util import CQKPInstance, CQKPSolution
from typing import Optional
from API_KEY import AMPLIFY_TOKEN


def naive_formulation(instance: CQKPInstance) -> Optional[CQKPSolution]:
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
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000    # 実行時間を 1000 ミリ秒に設定
    result = solve(f, client)

    val_lin = obj_lin.evaluate(result.best.values)
    val_qua = obj_qua.evaluate(result.best.values)
    obj_value = val_lin + val_qua

    XX = X.evaluate(result.best.values)
    Items = [i for i in range(N) if XX[i] > 0.5]
    
    result = CQKPSolution(Items, obj_value)
    return result
