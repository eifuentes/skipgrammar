"""
MovieLens Datasets and Helpers.

References:
- [MovieLens](https://grouplens.org/datasets/movielens/)
- [Lenskit datasets](https://github.com/lenskit/lkpy/blob/master/lenskit/datasets.py)
"""
import logging
import os

import numpy as np
import pandas as pd

from seqrep.datasets.common import cached

logger = logging.getLogger(__name__)

VARIANTS = {
    'ml-dev-small': {
        'origin': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'filename': 'ml-latest-small.zip',
        'extract': {
            'dirname': 'ml-latest-small'
        }
    },
    'ml-dev-large': {
        'origin': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
        'filename': 'ml-latest.zip',
        'extract': {
            'dirname': 'ml-latest'
        }
    },
    'ml-benchmark': {
        'origin': 'http://files.grouplens.org/datasets/movielens/ml-25m.zip',
        'filename': 'ml-25m.zip',
        'extract': {
            'dirname': 'ml-25m'
        }
    }
}


class MovieLens:
    """
    MovieLens datasets access class, including ml-20M, ml-latest, and ml-latest-small.
    Parameters:
        filepath (str): Filepath to the directory containing the dataset.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self._item_index_lu = MovieLens.fetch_movies(filepath).reset_index().set_index('item').to_dict().get('index')

    @cached
    def ratings(self):
        """
        The rating table.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        """

        filename = os.path.join(self.filepath, "ratings.csv")
        ratings = pd.read_csv(filename, dtype={"movieId": np.int32, "userId": np.int32, "rating": np.float64, "timestamp": np.int32})
        ratings.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        ratings["item"] = ratings.apply(lambda r: self._item_index_lu.get(r.loc['item']), axis='columns')
        ratings["timestamp"] = pd.to_datetime(ratings.timestamp, utc=True, unit='s', origin='unix')
        ratings = ratings.sort_values(['user', 'timestamp'], ascending=True)
        logger.debug("loaded %s, takes %d bytes", filename, ratings.memory_usage().sum())
        return ratings

    @staticmethod
    def fetch_movies(filepath):
        filename = os.path.join(filepath, "movies.csv")
        movies = pd.read_csv(filename, dtype={"movieId": np.int32, "title": np.object, "genres": np.object})
        movies.rename(columns={"movieId": "item"}, inplace=True)
        logger.debug("loaded %s, takes %d bytes", filename, movies.memory_usage().sum())
        return movies

    @cached
    def movies(self):
        """
        The movie table, with titles and genres. It is indexed by movie ID.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        """

        movies = self.fetch_movies(self.filepath)
        movies["item"] = movies.apply(lambda r: self._item_index_lu.get(r.loc['item']), axis='columns')
        movies.set_index("item", inplace=True)
        return movies

    @cached
    def ratings_movies(self):
        """
        The ratings and movie tables combined.
        """

        return pd.merge(
            left=self.ratings,
            right=self.movies,
            how='left',
            left_on='item',
            right_index=True
        )

    @cached
    def links(self):
        """
        The movie link table, connecting movie IDs to external identifiers. It is indexed by movie ID.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        """

        filename = os.path.join(self.filepath, "links.csv")
        links = pd.read_csv(filename, dtype={"movieId": np.int32, "imdbId": np.int64, "tmdbId": pd.Int64Dtype()})
        links.rename(columns={"movieId": "item"}, inplace=True)
        links["item"] = links.apply(lambda r: self._item_index_lu.get(r.loc['item']), axis='columns')
        links.set_index("item", inplace=True)
        logger.debug("loaded %s, takes %d bytes", filename, links.memory_usage().sum())
        return links

    @cached
    def tags(self):
        """
        The tag application table, recording user-supplied tags for movies.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        """

        filename = os.path.join(self.filepath, "tags.csv")
        tags = pd.read_csv(filename, dtype={"movieId": np.int32, "userId": np.int32, "tag": np.object, "timestamp": np.int32})
        tags.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        tags["item"] = tags.apply(lambda r: self._item_index_lu.get(r.loc['item']), axis='columns')
        tags["timestamp"] = pd.to_datetime(tags.timestamp, utc=True, unit='s', origin='unix')
        tags = tags.sort_values(['user', 'timestamp'], ascending=True)
        logger.debug("loaded %s, takes %d bytes", filename, tags.memory_usage().sum())
        return tags

    @cached
    def tag_genome(self):
        """
        The tag genome table, recording inferred item-tag relevance scores.
        This gets returned as a wide Pandas data frame, with rows indexed by item ID.
        >>> ml20m = MovieLens('data/ml-20m')
        """

        filename = os.path.join(self.filepath, "genome-scores.csv")
        tags = pd.read_csv(os.path.join(self.filepath, "genome-tags.csv"))
        tags["item"] = tags.apply(lambda r: self._item_index_lu.get(r.loc['item']), axis='columns')
        tags = tags.set_index("tagId")
        tags = tags["tag"].astype("category")
        genome = pd.read_csv(filename, dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float64})
        genome.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        genome["item"] = genome.apply(lambda r: self._item_index_lu.get(r.loc['item']), axis='columns')
        genome = genome.join(tags, on="tagId")
        genome = genome.pivot(index="item", columns="tag", values="relevance")
        logger.debug("loaded %s, takes %d bytes", filename, genome.memory_usage().sum())
        return genome
