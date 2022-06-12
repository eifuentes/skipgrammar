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

from skipgrammar.datasets.common import (UserItemIterableDataset, cached,
                                         get_file)

logger = logging.getLogger(__name__)

VARIANTS = {
    "ml-dev-small": {
        "origin": "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "filename": "ml-latest-small.zip",
        "extract": {"dirname": "ml-latest-small"},
    },
    "ml-dev-large": {
        "origin": "http://files.grouplens.org/datasets/movielens/ml-latest.zip",
        "filename": "ml-latest.zip",
        "extract": {"dirname": "ml-latest"},
    },
    "ml-benchmark": {
        "origin": "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "filename": "ml-25m.zip",
        "extract": {"dirname": "ml-25m"},
    },
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
        """

        filename = os.path.join(self.filepath, "ratings.csv")
        ratings = pd.read_csv(
            filename,
            dtype={
                "movieId": np.int32,
                "userId": np.int32,
                "rating": np.float64,
                "timestamp": np.int32,
            },
        )
        ratings.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        ratings["timestamp"] = pd.to_datetime(
            ratings.timestamp, utc=True, unit="s", origin="unix"
        )
        ratings = ratings.sort_values(["user", "timestamp"], ascending=True)
        logger.debug(
            "loaded %s, takes %d bytes", filename, ratings.memory_usage().sum()
        )
        return ratings

    @cached
    def movies(self):
        """
        The movie table, with titles and genres. It is indexed by movie ID.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        """

        filename = os.path.join(self.filepath, "movies.csv")
        movies = pd.read_csv(
            filename,
            dtype={"movieId": np.int32, "title": np.object, "genres": np.object},
        )
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
            how="left",
            left_on="item",
            right_index=True,
        )

    @cached
    def links(self):
        """
        The movie link table, connecting movie IDs to external identifiers. It is indexed by movie ID.
        >>> mlsmall = MovieLens('data/ml-latest-small')
        """

        filename = os.path.join(self.filepath, "links.csv")
        links = pd.read_csv(
            filename,
            dtype={"movieId": np.int32, "imdbId": np.int64, "tmdbId": pd.Int64Dtype()},
        )
        links.rename(columns={"movieId": "item"}, inplace=True)
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
        tags = pd.read_csv(
            filename,
            dtype={
                "movieId": np.int32,
                "userId": np.int32,
                "tag": np.object,
                "timestamp": np.int32,
            },
        )
        tags.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        tags["timestamp"] = pd.to_datetime(
            tags.timestamp, utc=True, unit="s", origin="unix"
        )
        tags = tags.sort_values(["user", "timestamp"], ascending=True)
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
        tags = tags.set_index("tagId")
        tags = tags["tag"].astype("category")
        genome = pd.read_csv(
            filename,
            dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float64},
        )
        genome.rename(columns={"userId": "user", "movieId": "item"}, inplace=True)
        genome = genome.join(tags, on="tagId")
        genome = genome.pivot(index="item", columns="tag", values="relevance")
        logger.debug("loaded %s, takes %d bytes", filename, genome.memory_usage().sum())
        return genome


class MovieLensUserItemDataset(UserItemIterableDataset):
    def __init__(
        self,
        variant,
        min_item_cnt_thresh=5,
        min_user_item_cnt_thresh=5,
        max_window_size_lr=10,
        max_sequence_length=20,
        shuffle=True,
    ):

        if variant not in VARIANTS.keys():
            raise KeyError(f"variant {variant} not supported.")

        dset_filepath = get_file(
            fname=VARIANTS.get(variant).get("filename"),
            origin=VARIANTS.get(variant).get("origin"),
            extract=True,
        )
        extract_dset_filepath = os.path.join(
            os.path.dirname(dset_filepath),
            VARIANTS.get(variant).get("extract").get("dirname"),
        )

        movielens_contnr = MovieLens(extract_dset_filepath)
        df = movielens_contnr.ratings
        logger.debug(len(df))

        item_cnts = df.item.value_counts()
        df = df.loc[
            df.item.isin(item_cnts.loc[item_cnts >= int(min_item_cnt_thresh)].index)
        ]
        user_cnts = df.user.value_counts()
        df = df.loc[
            df.user.isin(
                user_cnts.loc[user_cnts >= int(min_user_item_cnt_thresh)].index
            )
        ]
        logger.debug(len(df))

        item_df = (
            movielens_contnr.movies.reset_index()
            .reset_index()
            .rename(columns={"index": "id"})
        )
        item_df.id += 1
        item_id_lu = item_df.set_index("item").to_dict().get("id")
        item_df = item_df.drop("item", axis=1).set_index("id")
        df["id"] = df.item.map(item_id_lu)
        df = df.drop("item", axis=1)

        super().__init__(
            df,
            max_window_size_lr=max_window_size_lr,
            max_sequence_length=max_sequence_length,
            user_col="user",
            sort_col="timestamp",
            shuffle=shuffle,
        )

        self.id_metadata = item_df.to_dict(orient="index")
        self.num_items = len(self.id_metadata)

    def __iter__(self):
        return super().__iter__()
