import torch
from transformers import BertConfig, BertModel, BertTokenizer, BertForPreTraining, BatchEncoding
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

from notebooks.clutrr import Instance, Data as ClutrrData, clutrr_config


#%%

clutrr_data = ClutrrData(clutrr_config.train_path, clutrr_config.test_paths)

model = BertModel.from_pretrained(
    'bert-base-uncased', output_attentions=True, return_dict=True
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%%

# print(model)
# print(tokenizer)

raw_stories = [inst.raw_story for inst in clutrr_data.train[:100]]

inputs = tokenizer(raw_stories, return_tensors="pt", padding=True)

token_ids = inputs.data['input_ids'][0].tolist()
print(token_ids)
tokens_sample = tokenizer.convert_ids_to_tokens(token_ids)[:24]
print(tokens_sample)

#%%
outputs = model(**inputs)

#%%
attentions = outputs.attentions[0]
# todo: weighted avg?
attn_final = torch.mean(attentions, dim=1)

attn_sample = attn_final[0, ...]

for i, token in enumerate(tokens_sample):
    for j, other_token in enumerate(tokens_sample):
        attn = attn_sample[i, j]
        if attn > 0.04:
            print(f"{token} <-> {other_token}: {attn:.2f}")

# all attentions are
# torch.Size([1, 12, 24, 24])
# batch_size, num_heads, sequence_length, sequence_length


#%%

