from collections import defaultdict
from pprint import pprint
from typing import Union

import torch.nn
from matplotlib import pyplot as plt
import torch.nn.functional as F

from clutrr.preprocess import load_clutrr
from models.base import BaseNet
from models.factory import create_model
from results.store import store_results
from schemas import TrainConfig, TrainHistory, Experiment
from graph.schemas import InputBatch
from utils import transpose_2d_list


def run_experiment(experiment: Experiment):
    clutrr_data = load_clutrr()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_instances = (
        clutrr_data.train[:experiment.num_training_samples]
        if experiment.num_training_samples != -1 else clutrr_data.train
    )

    graph_factory = experiment.graph_factory_cls(
        entity_lst=clutrr_data.entity_lst,
        relation_lst=clutrr_data.relation_lst,
        batch_size=experiment.train_config.batch_size,
        device=device
    )
    train_data = graph_factory.create_batches(train_instances)

    test_data = {}
    for test_name in clutrr_data.test:
        test_instances = clutrr_data.test[test_name]
        test_batches = graph_factory.create_batches(test_instances)
        test_name = test_name.split("/")[-1]
        test_data[test_name] = test_batches

    model = create_model(
        model_type=experiment.model_type,
        num_nodes=graph_factory.input_dim,
        edge_dim=graph_factory.edge_dim,
        target_size=len(clutrr_data.relation_lst),
        device=device,
    )

    train_history = training_loop(model, experiment.train_config, train_data, test_data)

    show_plots(train_history, experiment.train_config)

    experiment_summary(experiment, train_history)

    store_results(experiment, train_history)


def training_loop(
    model: BaseNet,
    train_config: TrainConfig,
    train_data: list[InputBatch],
    test_data: dict[str, list[InputBatch]]
) -> TrainHistory:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)  # weight_decay=5e-4

    history: TrainHistory = {
        "train_losses": [],
        "train_acc": [],
        "test_acc": defaultdict(list),
    }

    for epoch in range(train_config.num_epochs):
        batch_loss = 0.
        correct = 0
        num_samples = 0
        model.train()

        for batch in train_data:
            logits = model(batch)

            y = batch.geo_batch.y.squeeze(1)
            loss = F.cross_entropy(logits, y, reduction='sum')
            batch_loss += loss.item()

            predictions = logits.max(dim=1)[1]
            correct += predictions.eq(y).sum().item()
            num_samples += predictions.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = batch_loss / num_samples
        history["train_losses"].append(avg_loss)
        train_accuracy = correct / num_samples
        history["train_acc"].append(train_accuracy)

        print(f'Epoch: {epoch:03d}, Train Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}')

        if epoch % train_config.evaluate_every == 0:
            print("testing")
            for index, test_name in enumerate(test_data):
                test_accuracy = test(test_data[test_name], model)
                history["test_acc"][test_name].append(test_accuracy)
                print(f'Epoch: {epoch:03d}, Test Set: {test_name}, Accuracy: {test_accuracy:.4f}')

    return history


def test(test_data: list[InputBatch], model: BaseNet) -> float:
    correct = 0
    total = 0
    model.eval()

    for test_batch in test_data:
        test_logits = model(test_batch)
        test_predictions = test_logits.max(dim=1)[1]
        test_true = test_batch.geo_batch.y.squeeze(1)
        correct += test_predictions.eq(test_true).sum().item()
        total += test_predictions.size(0)
    return correct / total


def show_plots(history: TrainHistory, train_config: TrainConfig) -> None:
    num_epochs = train_config.num_epochs

    def plot_variable(var: list, var_name: str, var_label: Union[str, list]):
        plt.ylabel(var_name)
        plt.xlabel("Epoch")

        x_marks = num_epochs // len(var) + (0 if num_epochs % len(var) == 0 else 1)
        x = torch.arange(0, num_epochs, x_marks)
        plt.xlim(0, num_epochs)

        plt.plot(x, var, label=var_label)
        plt.legend()
        plt.show()

    plot_variable(history["train_losses"], "Loss", "Average loss")
    plot_variable(history["train_acc"], "Train acc", "Train accuraccy")

    test_vars = [test_acc for test_acc in history["test_acc"].values()]
    test_labels = [test_name for test_name in history["test_acc"]]
    plot_variable(transpose_2d_list(test_vars), "Test acc", test_labels)


def experiment_summary(experiment: Experiment, train_history: TrainHistory) -> None:
    print("===========EXPERIMENT SUMMARY===========")
    pprint(experiment)
    print(f'Train loss: {min(train_history["train_losses"]):.4f}')
    print(f'Train accuracy: {max(train_history["train_acc"]):.4f}')
    for test_name, test_acc in train_history["test_acc"].items():
        print(f'Test Set: {test_name}, Accuracy: {max(test_acc):.4f}')
    print("========================================")
