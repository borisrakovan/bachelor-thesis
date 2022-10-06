from torch import Tensor, nn
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F

from graph.schemas import InputBatch
from models.base import BaseNet


class Bert(BaseNet):
    def __init__(self, target_size: int):
        super().__init__()
        # self.model = BertModel.from_pretrained(
        #     'bert-base-uncased', return_dict=True
        # )
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased", return_dict=True)

        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.lin1 = nn.Linear(in_features=768, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=target_size)

    def forward(self, input_batch: InputBatch) -> Tensor:
        raw_stories = [inst.bert_story for inst in input_batch.instances]

        inputs = self.tokenizer(raw_stories, return_tensors="pt", padding=True).to(self.model.device)

        # self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # We only care about DistilBERT's output for the [CLS] token,
        # which is located at index 0 of every encoded sequence.
        # Splicing out the [CLS] tokens gives us 2D data.

        last_hidden_state, = self.model(**inputs, return_dict=False)
        last_hidden_state = last_hidden_state[:, 0, :]

        # _, pooled_output = self.model(**inputs, return_dict=False)

        x = self.lin1(last_hidden_state)
        x = F.relu(x)
        x = self.lin2(x)
        return x
