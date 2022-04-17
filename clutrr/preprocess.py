"""
Disclaimer: Some of this code was significantly inspired by this repository:
https://github.com/uclnlp/ctp
"""

import yaml

import csv
import json

from collections import OrderedDict

from typing import List, Tuple, Any, Optional

from utils import path_to

_TEST_PATHS = [
    "data_089907f8/1.2_test.csv",
    "data_089907f8/1.3_test.csv",
    "data_089907f8/1.4_test.csv",
    "data_089907f8/1.5_test.csv",
    "data_089907f8/1.6_test.csv",
    "data_089907f8/1.7_test.csv",
    "data_089907f8/1.8_test.csv",
    "data_089907f8/1.9_test.csv",
    "data_089907f8/1.10_test.csv",
]


with open(path_to("relations_store.yaml"), 'r') as f:
    relations_dict = yaml.safe_load(f)

Fact = Tuple[str, str, str]
Story = List[Fact]


class Instance:
    def __init__(self,
                 story: Story,
                 target: Fact,
                 raw_story: str,
                 nb_nodes: Optional[int] = None):
        self._story = story
        self._target = target
        self._raw_story = raw_story
        self._nb_nodes = nb_nodes

    @property
    def story(self) -> Story:
        return self._story

    @property
    def target(self) -> Fact:
        return self._target

    @property
    def raw_story(self) -> str:
        return self._raw_story

    @property
    def nb_nodes(self) -> Optional[int]:
        return self._nb_nodes

    def __str__(self) -> str:
        return f'{self.story}\t{self.target}'


class Data:
    def __init__(self,
                 train_path,
                 test_paths: Optional[List[str]] = None,
                 with_tagged_entities: bool = False):
        self.relation_to_predicate = {r['rel']: k for k, v in relations_dict.items()
                                      for _, r in v.items() if k != 'no-relation'}

        self.relation_lst = sorted({r for r in self.relation_to_predicate.keys()})

        self.relation_to_idx = {rel: idx for idx, rel in enumerate(self.relation_lst)}

        self._train_instances = Data.parse(train_path, with_tagged_entities)
        entity_set = {s for i in self.train for (s, _, _) in i.story} | {o for i in self.train for (_, _, o) in i.story}

        self._test_instances = OrderedDict()
        for test_path in (test_paths if test_paths is not None else []):
            i_lst = self._test_instances[test_path] = Data.parse(test_path, with_tagged_entities)
            entity_set |= {s for i in i_lst for (s, _, _) in i.story} | {o for i in i_lst for (_, _, o) in i.story}

        self.entity_lst = sorted(entity_set)

        for instance in self.train:
            for s, r, o in instance.story:
                assert s in self.entity_lst and o in self.entity_lst
                assert r in self.relation_lst

    @property
    def train(self) -> List[Instance]:
        # TODO: tweak the number of tranining samples here
        return self._train_instances[:6000]

    @property
    def test(self) -> OrderedDict[str, List[Instance]]:
        return self._test_instances

    @staticmethod
    def _to_obj(s: str) -> Any:
        return json.loads(s.replace(")", "]").replace("(", "[").replace("'", "\""))

    @staticmethod
    def parse(path: str, with_tagged_entities: bool) -> List[Instance]:
        res = []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                _id, _, raw_story, query, _, target, _, _, _, _, _, story_edges, edge_types, _, genders, _, tmp, _ = row
                if len(_id) > 0:
                    nb_nodes = int(tmp[tmp.rfind(":") + 2:-1]) + 1
                    id_to_name = {i: name.split(':')[0] for i, name in enumerate(genders.split(','))}
                    _story, _edge, _query = Data._to_obj(story_edges), Data._to_obj(edge_types), Data._to_obj(query)
                    triples = [(id_to_name[s_id], p, id_to_name[o_id]) for (s_id, o_id), p in zip(_story, _edge)]
                    target = (_query[0], target, _query[1])
                    raw_story = raw_story if with_tagged_entities else raw_story.replace("[", "").replace("]", "")
                    instance = Instance(triples, target, raw_story, nb_nodes=nb_nodes)
                    res += [instance]
        return res


def load_clutrr() -> Data:
    return Data(
        train_path=path_to("data_db9b8f04/1.2,1.3,1.4_train.csv"),
        test_paths=[path_to(tf) for tf in _TEST_PATHS],
        with_tagged_entities=True
    )
