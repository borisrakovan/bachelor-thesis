import os
import pickle

from experiments.schemas import Experiment, TrainHistory
from utils import hash_string

dirname = os.path.dirname(__file__)

result_dir = os.path.join(dirname, "..", "data", "results")


def path_to(experiment: Experiment):
    fingerprint = hash_string(str(experiment))[:16]
    return os.path.join(result_dir, f"{fingerprint}.pickle")


def store_results(experiment: Experiment, train_history: TrainHistory) -> None:
    path = path_to(experiment)
    with open(path, "wb") as out:
        pickle.dump((experiment, train_history), out)

    print(f"Experiment results stored in {path}")


def load_results(experiment: Experiment, experiment_name: str) -> tuple[Experiment, TrainHistory]:
    if experiment_name == "text-bert":
        path = os.path.join(result_dir, "bert.pickle")
    elif experiment_name == "graph-gcn-com":
        path = os.path.join(result_dir, "97939af7db0202b4.pickle")
    else:
        path = path_to(experiment)
    with open(path, "rb") as inp:
        return pickle.load(inp)

#
# def regenerate():
#     for file in os.listdir(result_dir):
#         with open(file, "rb") as inp:
#             a, b = pickle.load(inp)
#
#         with open(file, "wb") as out:
#             pickle.dump((experiment, train_history), out)
