
# Learning complex natural language inferences with relational neural models

This repository contains code used to carry out different experiments for my bachelor thesis.

## Abstract

Large language models based on neural networks have recently achieved remarkable results across a wide range of NLP tasks. However, they still struggle with deeper understanding of semantics and reasoning about entities and their relationships. To address these shortcomings, in this thesis we examine alternative deep machine learning architectures with the intent of testing their logical reasoning and systematic generalization capabilities. We propose and implement several deep learning models, mainly from the deep relational learning category, and evaluate the proposed models on a textual benchmark selected from the NLU domain. In the empirical part, we manage to demonstrate the substantial performance gap between the standard NLU models that work with unstructured text data and more advanced models, mainly graph and recurrent neural networks, that allow to process more structured inputs. Finally, we propose suitable relational model biases to address the particular forms of relational reasoning in the selected benchmark and manage to achieve results comparable to state-of-the-art.


## Project structure

- `clutrr/` - code related to the preprocessing of CLUTRR dataset 
- `experiments/` - model training and evaluation of experiments  
- `graph/` - implementation of different graph construction methods
- `models/` - implementation of different deep neural networks
- `nlp/` - NLP-related utilities such as tokenization or word embeddings
- `notebooks/` - a collection of notebooks mainly exported from Google Colab
- `outputs/` - generated figures and plots used in the thesis
- `scripts/` - a miscellaneous collection of scripts for different purposes
- `main.py` - entrypoint for running experiments
- `utils.py` - miscellaneous utility functions
- `poetry.lock` & `pyproject.toml` - definition of project dependencies


clutrr/ & code related to the preprocessing of CLUTRR dataset \\ 
experiments/ & model training and evaluation of experiments \\  
graph/ & implementation of different graph construction methods \\
models/ & implementation of different deep neural networks \\
nlp/ & NLP&related utilities such as tokenization or word embeddings \\
notebooks/ & a collection of notebooks mainly exported from Google Colab \\
outputs/ & generated figures and plots used in the thesis \\
scripts/ & a miscellaneous collection of scripts for different purposes \\
main.py & entrypoint for running experiments \\
utils.py & miscellaneous utility functions \\
poetry.lock, pyproject.toml & definition of project dependencies \\
