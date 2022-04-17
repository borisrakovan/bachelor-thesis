from typing import TypeVar

import numpy as np

from hashlib import sha256


def hash_string(string: str):
    return sha256(string.encode('utf-8')).hexdigest()


def path_to(name):
    # return "/content/gdrive/MyDrive/bakalarka/data/clutrr/" + name
    return "/Users/boris.rakovan/Desktop/school/thesis/code/data/clutrr/" + name


def make_batches(size: int, batch_size: int) -> list[tuple[int, int]]:
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


T = TypeVar("T")


def transpose_2d_list(lst: list[list[T]]) -> list[list[T]]:
    return list(map(list, zip(*lst)))
