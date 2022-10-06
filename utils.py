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


def pad_sequences(sequences, max_len=None, dtype='int32', padding='post', truncating='post', value=0.):
    """
    Copied from some ML project

    Pads each sequence to the same length (length of the longest sequence).

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        max_len: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            max_len either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, max_len)

    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    lengths = [len(x) for x in sequences]

    num_samples = len(sequences)
    if max_len is None:
        max_len = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, max_len) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def print_latex_table(rows: list[list[str]]):
    for row in rows:
        vals = [str(val).replace("_", "\\_") for val in row]
        row_str = " & ".join(vals) + " \\\\"
        print(row_str)
        print("\\hline")

    print()

