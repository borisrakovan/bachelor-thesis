import os.path
import pickle

import numpy as np


EmbeddingDict = dict[str, np.array]


def path_to(name):
    # return "/content/gdrive/MyDrive/bakalarka/data/clutrr/" + name
    return "/Users/boris.rakovan/Desktop/school/thesis/code/data/" + name


class WordEmbeddings:
    def __init__(self):
        self.words, self.word_to_vec_map = self._read_embeddings(path_to("glove/glove.42B.300d.txt"))
        self.embedding_dim = len(self.word_to_vec_map["the"])
        self.special_ent_embedding = self.word_to_vec_map["person"]

    def _read_embeddings(self, path: str) -> tuple[list[str], EmbeddingDict]:
        emb_processed_path = path + "_cache"
        if not os.path.exists(emb_processed_path):
            embeddings = self._do_read_glove_embeddings(path)
            with open(emb_processed_path, "wb") as out:
                pickle.dump(embeddings, out)

        with open(emb_processed_path, "rb") as inp:
            return pickle.load(inp)

    def _do_read_glove_embeddings(self, path: str) -> tuple[list[str], EmbeddingDict]:
        words = []
        word_to_vec_map = dict()
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            while line is not None and line != "":
                values = line.split()
                word = values[0]
                vec = np.array([float(x) for x in values[1:]])
                words.append(word)
                word_to_vec_map[word] = vec
                line = f.readline()

        # return prepare_embeddings(words, word_to_vec_map)
        return words, word_to_vec_map

    # words, word_to_vec_map = read_glove_embeddings(path_to("glove.6B.100d.txt"))
