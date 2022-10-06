import time

from experiments.definitions import EXPERIMENT_DEFINITIONS
from experiments.runner import run_experiment


def main():
    start = time.time()

    experiment = EXPERIMENT_DEFINITIONS["graph-gcn-com"]

    print(f"Running experiment: {experiment}")
    run_experiment(experiment)

    end = time.time()
    print(f"Finished in {(end - start):.2f} seconds")


if __name__ == "__main__":
    main()
