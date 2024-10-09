from util import read_instance
from amplify_solver import naive_formulation
from pathlib import Path

if __name__ == '__main__':
    filepath = Path("./data/n50/d25_n50_inst1.json")
    instance = read_instance(filepath)
    solution = naive_formulation(instance)
    print(solution)