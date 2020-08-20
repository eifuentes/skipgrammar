# Seqrep
A framework for representing sequences as embeddings using machine learning.

## Models

### Skip-gram Negative Sampling (SGNS)

Popular natural language processing models such as `word2vec` and `bert` can be repurposed to learn relationships from arbitrary sequences of items. **Skip-gram Negative Sampling** is such an algorithm part of the `models` module. This is implemented in PyTorch components or can be composed as a PyTorch Lightning module. Both are availble under the relevent namespaces `seqrep.models.sgns` and `seqrep.models.lighting.sgns`.

## Datasets

### MovieLens

The popular recommendation system dataset [MovieLens](https://grouplens.org/datasets/movielens/) is availble in three variants via the `dataset` module.
