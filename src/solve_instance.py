from util import read_instance, CQKPSolution, CQKPInstance
from amplify_solver import formulation_qubo_card, formulation_unary, formulation_binary, evaluate_objective
from amplify import FixstarsClient, solve
from API_KEY import AMPLIFY_TOKEN
from pathlib import Path


def example_qubo_card(instance: CQKPInstance) -> None:
    # single
    f, X = formulation_qubo_card(instance=instance, lambda_card=500)
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000
    result = solve(f, client)

    # decode
    XX = X.evaluate(result.best.values)
    N = instance.N
    res = evaluate_objective(instance, XX)
    Items = [i for i in range(N) if XX[i] > 0.5]
    result = CQKPSolution(Items, res["obj"], res["card"], res["cap"])
    print(result)
    return


def example_unary(instance: CQKPInstance) -> None:
    # single
    f, X, S = formulation_unary(instance=instance, lambda_card=200)
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000
    result = solve(f, client)

    # decode
    XX = X.evaluate(result.best.values)
    SS = S.evaluate(result.best.values)
    N = instance.N
    res = evaluate_objective(instance, XX)
    Items = [i for i in range(N) if XX[i] > 0.5]
    print(XX)
    print(SS)
    result = CQKPSolution(Items, res["obj"], res["card"], res["cap"])
    print(result)


if __name__ == "__main__":
    filepath = Path("./data/n50/d25_n50_inst1.json")
    instance = read_instance(filepath)

    # single
    f, X, S = formulation_binary(instance=instance, lambda_card=200)
    client = FixstarsClient()
    client.token = AMPLIFY_TOKEN
    client.parameters.timeout = 1000
    result = solve(f, client)
    print(result.best)