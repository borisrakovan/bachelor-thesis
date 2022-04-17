from dataclasses import dataclass

import spacy
import torch
from torch.autograd import Function
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch_geometric.nn import GCNConv

from notebooks.clutrr import Instance, Data as ClutrrData, clutrr_config
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from torch_geometric.data import Data, Batch as GeoBatch
from torch_geometric import nn as geo_nn


#%%

clutrr_data = ClutrrData(clutrr_config.train_path, clutrr_config.test_paths, with_tagged_entities=True)


def path_to(name):
    # return "/content/gdrive/MyDrive/bakalarka/data/clutrr/" + name
    return "/Users/boris.rakovan/Desktop/school/thesis/code/data/" + name

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_glove_embeddings(path: str) -> tuple[list[str], dict[str, list[float]]]:
    words = []
    word_to_vec_map = dict()
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        while line is not None and line != "":
            values = line.split()
            word = values[0]
            vec = [float(x) for x in values[1:]]
            words.append(word)
            word_to_vec_map[word] = vec
            line = f.readline()

    return words, word_to_vec_map


words, word_to_vec_map = read_glove_embeddings(path_to("glove.6B.100d.txt"))
embedding_dim = len(word_to_vec_map["the"])

# todo: is this sensible?
SPECIAL_ENTITY_EMBEDDING = word_to_vec_map["person"]


#%%

nlp = spacy.load("en_core_web_sm")


def make_batches(size: int, batch_size: int) -> list[tuple[int, int]]:
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


@dataclass
class InputBatch:
    x: Tensor  # (batch_size, seq_len, emb_dim)
    attn_mask: Tensor  # (batch_size, seq_len, seq_len)
    y: Tensor  # (batch_size, 1)
    targets: Tensor  # (batch_size, 2)


def tokenize_normalize(instance: Instance, vocab: list[str]) -> tuple[list[str], set[str]]:
    doc = nlp(instance.raw_story)
    normalized_tokens = []
    entities = []
    reading_entity = False

    sanity_check = 0

    for token in doc:
        if token.text in "[]":
            # toggle
            reading_entity = not reading_entity
            sanity_check += 1
            continue
        if token.is_alpha:
            normalized = token.lower_.strip()
            if normalized in vocab or reading_entity:
                # append only tokens for which embeddings exist
                normalized_tokens.append(normalized)
            if reading_entity:
                entities.append(normalized)

    if not sanity_check == 2 * len(entities):
        assert False

    return normalized_tokens, set(entities)


def encode_batch(instances: list[Instance]) -> InputBatch:
    batch_size = len(instances)
    processed_stories = [tokenize_normalize(inst, vocab=words) for inst in instances]

    max_story_len = max(len(x[0]) for x in processed_stories)

    x = []
    attn_mask = torch.ones((batch_size, max_story_len, max_story_len), dtype=torch.bool)
    batch_targets = []
    y = []

    for story_index, (story_tokens, story_entities) in enumerate(processed_stories):
        current_instance = instances[story_index]
        story_encoding: list[np.array] = []
        for token_index, token in enumerate(story_tokens):
            if token in story_entities:
                token_emb = SPECIAL_ENTITY_EMBEDDING
            else:
                token_emb = word_to_vec_map.get(token)
                if not token_emb:
                    print(f"Missing embedding: {token}. Shouldn't happen.")
                    raise KeyError

            story_encoding.append(token_emb)

        padding = [0] * embedding_dim
        story_len = len(story_encoding)
        for _ in range(max_story_len - story_len):
            story_encoding.append(padding)

        x.append(story_encoding)
        attn_mask[story_index, 0:story_len, 0:story_len] = torch.zeros((story_len, story_len), dtype=torch.bool)

        tgt_frm, tgt_rel, tgt_to = current_instance.target
        try:
            target_edge = [
                next(idx for idx, token in enumerate(story_tokens) if token == tgt_frm.lower()),
                next(idx for idx, token in enumerate(story_tokens) if token == tgt_to.lower())
            ]
        except StopIteration:
            print(story_tokens)
            print(story_entities)
            print(current_instance.raw_story)
            print(current_instance.target)
            raise
        batch_targets.append(target_edge)
        y.append(clutrr_data.relation_to_idx[tgt_rel])

    return InputBatch(
        x=torch.tensor(x),
        attn_mask=attn_mask,
        y=torch.tensor(y, dtype=torch.long, device=device).view(-1, 1),
        targets=torch.tensor(batch_targets, dtype=torch.long, device=device)
    )


def to_batches(
    instances: list[Instance], batch_size: int
) -> list[InputBatch]:

    nb_instances = len(instances)
    batches = make_batches(nb_instances, batch_size)
    res = []

    for i, (batch_start, batch_end) in enumerate(batches):
        if i % 10 == 0:
            print(f"Processed {i}/{len(batches)}")
        batch_instances = instances[batch_start:batch_end]
        res.append(encode_batch(batch_instances))
    return res


cluttr_data_train = clutrr_data.train[:100]
train_data = to_batches(instances=cluttr_data_train, batch_size=25)

#%%

class AttentionTextToGraphNetwork(nn.Module):
    def __init__(self, qkv_embed_dim: int):
        super(AttentionTextToGraphNetwork, self).__init__()
        factory_kwargs = {'device': device, 'dtype': torch.long}

        self.q_proj_weight = Parameter(torch.empty((qkv_embed_dim, qkv_embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((qkv_embed_dim, qkv_embed_dim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((qkv_embed_dim, qkv_embed_dim), **factory_kwargs))

        # todo bias?

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        # todo bias?
    def forward(self):
        pass

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


#%%

# print(train_data)
print(train_data[0].x.shape)

#%%


class GCN(torch.nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)

        self.target_size = len(clutrr_data.relation_lst)
        # todo try embed_dim * 3 with averaged graph emb
        self.linear = nn.Linear(embed_dim * 2, self.target_size)

    def forward(self, batch, target: Tensor) -> Tensor:
        # target.shape == (batch_size, 2)

        edge_index = batch.edge_index

        x = self.conv1(batch.x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        batch_size = torch.max(batch.batch).item() + 1
        query_embs = torch.zeros((batch_size, self.embed_dim * 2))

        for index in range(batch_size):
            graph_emb = x[batch.batch == index, :]
            query_edge = target[index, :]
            query_emb = [graph_emb[query_edge[0]], graph_emb[query_edge[1]]]
            query_emb = torch.cat(query_emb, dim=0)
            query_embs[index, :] = query_emb

        logits = self.linear(query_embs)

        return logits


class Mask(Function):
    @staticmethod
    def forward(ctx, attn: Tensor, attn_cutoff: int):
        ctx.save_for_backward(attn)

        ctx.attn_cutoff = attn_cutoff
        mask = attn >= attn_cutoff
        edge_index = torch.nonzero(mask)

        return edge_index

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None

        attn = ctx.saved_tensors
        grad_attn = None

        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None


class Model(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)

        self.gcn = GCN(embed_dim=embed_dim).to(device)

        self.attn_cutoff = 0.05

    def forward(self, x: Tensor, attn_mask: Tensor, y: Tensor, targets: Tensor):
        # todo add more attn layers - if it makes sense
        attn_out, attn = self.attention(x, x, x, attn_mask=attn_mask)

        # take attn and selectively create the input graph

        # edge_mask = attn >= self.attn_cutoff
        # naive approach (non-differentiable?)

        # edges = torch.tensor(, requires_grad=True)
        # x is always the same

        # todo: try "deleting" edges using x instead of edge index
        #  set given nodes to 0s
        batch_data = []

        for index in range(x.size(0)):
            x_geo = x[index, :, :]  # (seq_len, emb_dim)
            y_geo = y[index, :]

            edge_index = Mask.apply(attn[index, :, :], self.attn_cutoff).t().contiguous()

            # edge_index = torch.nonzero(edge_mask[index, :, :]).t().contiguous()

            # edge_index = torch.masked_select()
            # edges_x = torch.arange(0, 4).repeat(4, 1)
            # edges_y = torch.arange(0, 4).repeat(4, 1).t()

            # edges = torch.stack((edges_y, edges_x), dim=2)
            # edge_attr = []


            # for i in range(edge_mask.size(1)):
            #     for j in range(edge_mask.size(1)):
            #         if edge_mask[index, i, j]:
            #             edge_index_t.append([i, j])
            #             # todo use this?
            #             edge_attr.append(attn[index, i, j])
            # edge_index = [[edge[0] for edge in edge_index_t], [edge[1] for edge in edge_index_t]]

            attn_tmp = attn_mask[index, 0, :]
            real_seq_len = attn_tmp[attn_tmp == False].size(0)
            print(f"Fully connected: {real_seq_len ** 2}. Pruned: {edge_index.size(1)}")

            batch_data.append(
                Data(
                    x=x_geo,
                    edge_index=edge_index,
                    # edge_attr=torch.tensor(edge_attr, dtype=torch.long, device=device).view(-1, 1),
                    y=y_geo
                )
            )

        geo_batch = GeoBatch.from_data_list(batch_data)

        return self.gcn(geo_batch, targets), attn


#%%

model = Model(embed_dim=embedding_dim).to(device)

#%%

print(list(model.parameters()))

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # weight_decay=5e-4
evaluate_every = 10
num_epochs = 10


def test(test_set) -> float:
    correct = 0
    model.eval()

    for test_batch, test_target in test_set:
        test_logits = model(test_batch, test_target)
        test_predictions = test_logits.max(dim=1)[1]
        correct += test_predictions.eq(test_batch.y).sum().item()
    return correct / len(test_set)


average_losses = []

for epoch in range(num_epochs):
    batch_loss = 0.
    correct = 0
    model.train()

    for batch in train_data:
        logits, attn = model(batch.x, batch.attn_mask, batch.y, batch.targets)
        attn_np = attn[-1].detach().numpy()

        # see whether the weights are being learnt
        # print(model.gcn.conv1.lin.weight.sum())
        # print(model.attention.in_proj_weight[1].sum())

        y = batch.y.squeeze(1)
        loss = F.cross_entropy(logits, y, reduction='sum')
        batch_loss += loss.item()

        predictions = logits.max(dim=1)[1]
        correct += predictions.eq(y).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # assert False

    avg_loss = batch_loss / len(cluttr_data_train)
    average_losses.append(avg_loss)
    train_accuracy = correct / len(cluttr_data_train)

    print(f'Epoch: {epoch:03d}, Train Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}')

    # if epoch % evaluate_every == 0:
    #     print("testing")
    #     for name in test_data:
    #         test_accuracy = test(test_data[name])
    #         print(f'Epoch: {epoch:03d}, Test Set: {name.rsplit("/")[-1]}, Accuracy: {test_accuracy:.4f}')

