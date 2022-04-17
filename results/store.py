import os
import pickle

from schemas import Experiment, TrainHistory
from utils import hash_string

dirname = os.path.dirname(__file__)


def path_to(experiment: Experiment):
    fingerprint = hash_string(str(experiment))[:16]
    return os.path.join(dirname, "..", "data", "results", f"{fingerprint}.pickle")


def store_results(experiment: Experiment, train_history: TrainHistory) -> None:
    path = path_to(experiment)
    with open(path, "wb") as out:
        pickle.dump((experiment, train_history), out)

    print(f"Experiment results stored in {path}")


def load_results(experiment: Experiment) -> tuple[Experiment, TrainHistory]:
    with open(path_to(experiment), "rb") as inp:
        return pickle.load(inp)
