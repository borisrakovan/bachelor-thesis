from experiments.schemas import Experiment, TrainConfig
from graph.dependency_graph import DependencyGraph
from graph.kinship_graph import KinshipGraph
from graph.complete_graph import CompleteGraphV3
from models.factory import ModelType

EXPERIMENT_NAMES = [
    "text-ff-base",
    "text-ff-pos",
    "text-bert",
    "graph-gcn-com",
    "graph-gcn-dep",
    "graph-gcn-kin",
    "graph-egcn-dep",
    "graph-egcn-kin",
    "graph-egat-dep",
    "graph-egat-kin",
    "seq-rnn",
    "seq-lstm",
]


def category_names(cat: str):
    return [
        name for name in EXPERIMENT_NAMES if name.startswith(cat)
    ]


EXPERIMENT_DEFINITIONS: dict[str, Experiment] = {
    # -- text-based --
    "text-ff-base":
        Experiment(
            model_type=ModelType.FEED_FORWARD,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    "text-ff-pos":
        Experiment(
            model_type=ModelType.FEED_FORWARD_POS,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=60,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    "text-bert":
        Experiment(
            model_type=ModelType.BERT,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=64,
                num_epochs=20,
                lr=0.001,
            ),
        ),
    # -- graph-based --
    "graph-gcn-com":
        Experiment(
            model_type=ModelType.GCN_BASELINE,
            graph_factory_cls=CompleteGraphV3,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    "graph-gcn-dep":
        Experiment(
            model_type=ModelType.GCN_BASELINE_EMB,
            graph_factory_cls=DependencyGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    "graph-gcn-kin":
        Experiment(
            model_type=ModelType.GCN_BASELINE_EMB,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    "graph-egcn-dep":
        Experiment(
            model_type=ModelType.EDGE_GCN,
            graph_factory_cls=DependencyGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=40,
                lr=0.001,
            ),
        ),
    "graph-egcn-kin":
        Experiment(
            model_type=ModelType.EDGE_GCN,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    "graph-egat-dep":
        Experiment(
            model_type=ModelType.EDGE_GAT,
            graph_factory_cls=DependencyGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=32,
                lr=0.0015,
            ),
        ),
    "graph-egat-kin":
        Experiment(
            model_type=ModelType.EDGE_GAT,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
    # -- sequence-based --
    "seq-rnn":
        Experiment(
            model_type=ModelType.LSTM,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=40,
                lr=0.001,
            ),
        ),
    "seq-lstm":
        Experiment(
            model_type=ModelType.LSTM,
            graph_factory_cls=KinshipGraph,
            train_config=TrainConfig(
                batch_size=100,
                num_epochs=64,
                lr=0.001,
            ),
        ),
}

assert EXPERIMENT_NAMES == list(EXPERIMENT_DEFINITIONS.keys())
