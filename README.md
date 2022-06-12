# Skip-Grammar
A framework for representing sequences as embeddings.

## Models

### Skip-gram Negative Sampling (SGNS)

Popular natural language processing models such as `word2vec` and `bert` can be repurposed to learn relationships from arbitrary sequences of items. **Skip-gram Negative Sampling** is such an algorithm part of the `models` module. This is implemented in PyTorch components or can be composed as a PyTorch Lightning module. Both are availble under the relevent namespaces `skipgrammar.models.sgns` and `skipgrammar.models.lighting.sgns`.

## Datasets

### Last.FM

The [Last.FM Dataset-1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) dataset is comprised of the listening history of approximately 1,000 users from the music service [Last.FM](https://www.last.fm/). The dataset is availble at the project's main site [here](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) and also preprocessed [here](https://github.com/eifuentes/lastfm-dataset-1K) for ease of use. The variants in the `dataset` module use the latter.

### MovieLens

The popular recommendation system dataset [MovieLens](https://grouplens.org/datasets/movielens/) is availble in three variants via the `dataset` module.
