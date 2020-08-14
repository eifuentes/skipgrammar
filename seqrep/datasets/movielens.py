"""
MovieLens Datasets[1] and Helpers.

Borrowed heavily:
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

    @cached
    def ratings(self):
        """
        The rating table.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.ratings
                user  item  rating   timestamp
        0          1    31     2.5  1260759144
        1          1  1029     3.0  1260759179
        2          1  1061     3.0  1260759182
        3          1  1129     2.0  1260759185
        4          1  1172     4.0  1260759205
        ...
        [100004 rows x 4 columns]
        """

        filename = os.path.join(self.filepath, "ratings.csv")
        ratings = pd.read_csv(filename, dtype={"movieId": np.int32, "userId": np.int32, "rating": np.float64, "timestamp": np.int32})
        ratings.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        logger.debug("loaded %s, takes %d bytes", filename, ratings.memory_usage().sum())
        return ratings

    @cached
    def movies(self):
        """
        The movie table, with titles and genres. It is indexed by movie ID.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.movies
                                                            title                                           genres
        item
        1                                        Toy Story (1995)      Adventure|Animation|Children|Comedy|Fantasy
        2                                          Jumanji (1995)                       Adventure|Children|Fantasy
        3                                 Grumpier Old Men (1995)                                   Comedy|Romance
        4                                Waiting to Exhale (1995)                             Comedy|Drama|Romance
        5                      Father of the Bride Part II (1995)                                           Comedy
        ...
        [9125 rows x 2 columns]
        """

        filename = os.path.join(self.filepath, "movies.csv")
        movies = pd.read_csv(filename, dtype={"movieId": np.int32, "title": np.object, "genres": np.object})
        movies.rename(columns={"movieId": "item"}, inplace=True)
        movies.set_index("item", inplace=True)
        logger.debug("loaded %s, takes %d bytes", filename, movies.memory_usage().sum())
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
        >>> mlsmall.links
                 imdbId  tmdbId
        item
        1        114709     862
        2        113497    8844
        3        113228   15602
        4        114885   31357
        5        113041   11862
        ...
        [9125 rows x 2 columns]
        """

        filename = os.path.join(self.filepath, "links.csv")
        links = pd.read_csv(filename, dtype={"movieId": np.int32, "imdbId": np.int64, "tmdbId": pd.Int64Dtype()})
        links.rename(columns={"movieId": "item"}, inplace=True)
        links.set_index("item", inplace=True)
        logger.debug("loaded %s, takes %d bytes", filename, links.memory_usage().sum())
        return links

    @cached
    def tags(self):
        """
        The tag application table, recording user-supplied tags for movies.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        >>> mlsmall.tags
              user  ...   timestamp
        0       15  ...  1138537770
        1       15  ...  1193435061
        2       15  ...  1170560997
        3       15  ...  1170626366
        4       15  ...  1141391765
        ...
        [1296 rows x 4 columns]
        """

        filename = os.path.join(self.filepath, "tags.csv")
        tags = pd.read_csv(filename, dtype={"movieId": np.int32, "userId": np.int32, "tag": np.object, "timestamp": np.int32})
        tags.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        logger.debug("loaded %s, takes %d bytes", filename, tags.memory_usage().sum())
        return tags

    @cached
    def tag_genome(self):
        """
        The tag genome table, recording inferred item-tag relevance scores.  This gets returned
        as a wide Pandas data frame, with rows indexed by item ID.
        >>> ml20m = MovieLens('data/ml-20m')
        >>> ml20m.tag_genome
        tag         007  007 (series)  18th century  ...     wwii   zombie  zombies
        item                                         ...
        1       0.02500       0.02500       0.05775  ...  0.03625  0.07775  0.02300
        2       0.03975       0.04375       0.03775  ...  0.01475  0.09025  0.01875
        3       0.04350       0.05475       0.02800  ...  0.01950  0.09700  0.01850
        4       0.03725       0.03950       0.03675  ...  0.01525  0.06450  0.01300
        5       0.04200       0.05275       0.05925  ...  0.01675  0.10750  0.01825
        ...
        [10381 rows x 1128 columns]
        """

        filename = os.path.join(self.filepath, "genome-scores.csv")
        tags = pd.read_csv(os.path.join(self.filepath, "genome-tags.csv"))
        tags = tags.set_index("tagId")
        tags = tags["tag"].astype("category")
        genome = pd.read_csv(filename, dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float64})
        genome.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        genome = genome.join(tags, on="tagId")
        genome = genome.pivot(index="item", columns="tag", values="relevance")
        logger.debug("loaded %s, takes %d bytes", filename, genome.memory_usage().sum())
        return genome
