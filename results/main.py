from graph.dependency_graph import DependencyGraph
from graph.kinship_graph import KinshipGraph
from graph.total_graph import TotalGraphV1, TotalGraphV2
from models.factory import ModelType
from results.plots import plot_multiline_series
from results.store import load_results
from schemas import TrainConfig, Experiment
from utils import transpose_2d_list


def evaluate_experiments(experiments: dict[str, Experiment]) -> None:
    y = []
    labels = []
    for exp_name in experiments:
        try:
            _, train_history = load_results(experiments[exp_name])
        except FileNotFoundError:
            raise Exception(f"Missing results for experiment {experiments[exp_name]}")

        test_acc_histories = train_history["test_acc"]
        test_accuracies = []
        for test_name in test_acc_histories:
            best_test_acc = max(test_acc_histories[test_name])
            test_accuracies.append(best_test_acc)
        y.append(test_accuracies)
        labels.append(exp_name)

    relation_lens = list(range(2, 10 + 1))
    assert len(relation_lens) == len(y[0])

    plot_multiline_series(
        x=relation_lens,
        y=transpose_2d_list(y),
        labels=labels,
        title="Systematic generalization of different models",
        x_label="Relation length",
        y_label="Accuracy",
    )


def evaluate_model_results() -> None:
    train_config = TrainConfig(
        batch_size=32,
        num_epochs=64,
        lr=0.001,
        evaluate_every=10
    )

    model_types = {
        "EdgeGCN": ModelType.EDGE_GCN,
        "EdgeGAT": ModelType.EDGE_GAT,
    }

    experiments = {
        exp_name:
            Experiment(
                model_type=model_type,
                graph_factory_cls=KinshipGraph,
                train_config=train_config,
                # num_training_samples=100,
            )
        for exp_name, model_type in model_types.items()
    }

    evaluate_experiments(experiments)


def evaluate_graph_results() -> None:
    train_config = TrainConfig(
        batch_size=32,
        num_epochs=64,
        lr=0.001,
        evaluate_every=10
    )

    experiment_setups = {
        "Dependency graph": (DependencyGraph, ModelType.EDGE_GAT),
        "Total graph V1": (TotalGraphV1, ModelType.GCN_MP), # todo
        "Total graph V2": (TotalGraphV2, ModelType.GCN_MP),
        "Kinship graph": (KinshipGraph, ModelType.EDGE_GAT),
    }

    experiments = {
        exp_name:
            Experiment(
                model_type=exp_setup[1],
                graph_factory_cls=exp_setup[0],
                train_config=train_config,
                # num_training_samples=100,
            )
        for exp_name, exp_setup in experiment_setups.items()
    }

    evaluate_experiments(experiments)


def main():
    evaluate_model_results()


if __name__ == "__main__":
    main()