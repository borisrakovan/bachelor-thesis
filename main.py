import time

from experiment import run_experiment, Experiment, TrainConfig
from graph.dependency_graph import DependencyGraph
from graph.kinship_graph import KinshipGraph
from models.factory import ModelType


def main():
    start = time.time()

    train_config = TrainConfig(
        batch_size=32,
        num_epochs=64,
        lr=0.001,
        evaluate_every=10
    )

    experiment = Experiment(
        model_type=ModelType.EDGE_GAT,
        graph_factory_cls=KinshipGraph,
        train_config=train_config,
        num_training_samples=-1,
    )

    print("Running experiment:")
    print(experiment)

    run_experiment(experiment)

    end = time.time()
    print(f"Finished in {(end - start):.2f} seconds")


if __name__ == "__main__":
    main()