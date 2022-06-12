"""
Last.FM Datasets and Helpers.

References:
- [Last.FM Dataset 1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
- [Lenskit datasets](https://github.com/lenskit/lkpy/blob/master/lenskit/datasets.py)
"""
import logging
import os

import pandas as pd

from skipgrammar.datasets.common import (UserItemIterableDataset, cached,
                                         get_file)

logger = logging.getLogger(__name__)

VARIANTS = {
    "lastfm-50": {
        "origin": "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-50.snappy.parquet",
        "filename": "lastfm-dataset-50.snappy.parquet",
    },
    "lastfm-1k": {
        "origin": "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-1k.snappy.parquet",
        "filename": "lastfm-dataset-1k.snappy.parquet",
    },
}

USER_PROFILE = {
    "origin": "https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/userid-profile.tsv.zip",
    "filename": "userid-profile.tsv.zip",
    "extract": {"filename": "userid-profile.tsv"},
}


class LastFM:
    """
    Last.FM datasets access class, including lastmf-50 and lastfm-1k.
    Parameters:
        listens_filepath (str): Filepath to the parquet file containing the user listening history dataset.
        user_profile_filepath (str): Filepath to the tab seperated (.tsv) file containing the user profile dataset.
    """

    def __init__(self, listens_filepath, user_profile_filepath):
        self.listens_filepath = listens_filepath
        self.user_profile_filepath = user_profile_filepath

    @cached
    def listens(self):
        """
        The listens table.
        >>> lfdset = LastFM(listens_filepath='data/lastfm-dataset-50.snappy.parquet', ...)
        """
        listens = pd.read_parquet(self.listens_filepath)
        logger.debug(
            "loaded %s, takes %d bytes",
            self.listens_filepath,
            listens.memory_usage().sum(),
        )
        return listens

    @cached
    def users(self):
        """
        The user profile table. It is indexed by user ID.
        >>> lfdset = LastFM(user_profile_filepath='data/userid-profile.tsv', ...)
        """

        users = (
            pd.read_csv(self.user_profile_filepath, sep="\t")
            .rename(columns={"#id": "user_id"})
            .set_index("user_id")
            .sort_index()
        )
        users["registered"] = pd.to_datetime(users.registered, utc=True)
        logger.debug(
            "loaded %s, takes %d bytes",
            self.user_profile_filepath,
            users.memory_usage().sum(),
        )
        return users


class LastFMUserItemDataset(UserItemIterableDataset):
    def __init__(
        self,
        variant,
        min_item_cnt_thresh=3,
        min_user_item_cnt_thresh=5,
        subsample_thresh=1e-5,
        item_dist_exp=0.75,
        session_timedelta="800s",
        max_window_size_lr=10,
        max_sequence_length=20,
        shuffle=True,
    ):

        if variant not in VARIANTS.keys():
            raise KeyError(f"last.fm variant {variant} not supported.")

        dset_filepath = get_file(
            fname=VARIANTS.get(variant).get("filename"),
            origin=VARIANTS.get(variant).get("origin"),
            extract=False,
        )
        user_dset_filepath = get_file(
            fname=USER_PROFILE.get("filename"),
            origin=USER_PROFILE.get("origin"),
            extract=True,
        )
        user_extract_dset_filepath = os.path.join(
            os.path.dirname(user_dset_filepath),
            USER_PROFILE.get("extract").get("filename"),
        )

        lfdset = LastFM(dset_filepath, user_extract_dset_filepath)

        df = lfdset.listens
        logger.debug(len(df))

        item_cnts = df.artist_name.value_counts()
        df = df.loc[
            df.artist_name.isin(
                item_cnts.loc[item_cnts >= int(min_item_cnt_thresh)].index
            )
        ]
        user_cnts = df.user_id.value_counts()
        df = df.loc[
            df.user_id.isin(
                user_cnts.loc[user_cnts >= int(min_user_item_cnt_thresh)].index
            )
        ]
        artists = self.subsample(df, item_col="artist_name", thresh=subsample_thresh)
        df = df.loc[df.artist_name.isin(artists)]
        logger.debug(len(df))

        item_id_lu = {
            a: i
            for i, a in zip(
                range(1, df.artist_name.nunique() + 1),
                df.artist_name.sort_values().unique(),
            )
        }
        df["artist_cd"] = df.artist_name.map(item_id_lu)

        super().__init__(
            df,
            max_window_size_lr=max_window_size_lr,
            max_sequence_length=max_sequence_length,
            user_col="user_id",
            sort_col="timestamp",
            item_col="artist_cd",
            session_timedelta=session_timedelta,
            shuffle=shuffle,
        )

        self.id_metadata = {i: a for a, i, in item_id_lu.items()}
        self.num_items = len(self.id_metadata)
        self.item_dist = self.item_distribution(
            self.df, item_col="artist_cd", p=item_dist_exp
        )

    def __iter__(self):
        return super().__iter__()
