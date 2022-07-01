"""
Common utilities for datasets.

References:
- [Keras utils](https://github.com/keras-team/keras/tree/34231971fa47cb2477b357c1a368978de4128294/keras/utils)
- [Lenskit datasets](https://github.com/lenskit/lkpy/blob/master/lenskit/datasets.py)
"""
import collections
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import time
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from torch.utils.data import Dataset as MapDataset
from torch.utils.data import IterableDataset, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(
        self, target, width=30, verbose=1, interval=0.05, stateful_metrics=None
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        ) or "ipykernel" in sys.modules
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [
                        v * (current - self._seen_so_far),
                        current - self._seen_so_far,
                    ]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += current - self._seen_so_far
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = " - %.0fs" % (now - self._start)
        if self.verbose == 1:
            if (
                now - self._last_update < self.interval
                and self.target is not None
                and current < self.target
            ):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write("\b" * prev_total_width)
                sys.stdout.write("\r")
            else:
                sys.stdout.write("\n")

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = "%%%dd/%d [" % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = " - ETA: %s" % eta_format
            else:
                if time_per_unit >= 1:
                    info += " %.0fs/step" % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += " %.0fms/step" % (time_per_unit * 1e3)
                else:
                    info += " %.0fus/step" % (time_per_unit * 1e6)

            for k in self._values:
                info += " - %s:" % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                else:
                    info += " %s" % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if self.target is not None and current >= self.target:
                info += "\n"

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += " - %s:" % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                info += "\n"

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
    # Returns
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    # Example
    ```python
        >>> from keras.data_utils import _hash_file
        >>> _hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        The file hash
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        Whether the file is valid
    """
    if (algorithm == "sha256") or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def get_file(
    fname,
    origin,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
):
    """Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.skipgrammar`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.skipgrammar/datasets/example.txt`.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.
    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Seqrep cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the Seqrep directory.
    # Returns
        Path to the downloaded file
    """  # noqa
    if cache_dir is None:
        if "SEQREP_HOME" in os.environ:
            cache_dir = os.environ.get("SEQREP_HOME")
        else:
            if os.access(os.path.expanduser("~"), os.W_OK):
                os.makedirs(
                    os.path.join(os.path.expanduser("~"), ".skipgrammar"), exist_ok=True
                )
            cache_dir = os.path.join(os.path.expanduser("~"), ".skipgrammar")
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".skipgrammar")
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + ".tar.gz"
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated because the "
                    + hash_algorithm
                    + " file hash does not match the original value of "
                    + file_hash
                    + " so we will re-download the data."
                )
                download = True
    else:
        download = True

    if download:
        print("Downloading data from", origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = "URL fetch failure on {} : {} -- {}"
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format="tar")
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def cached(prop):
    cache = "_cached_" + prop.__name__

    def getter(self):
        val = getattr(self, cache, None)
        if val is None:
            val = prop(self)
            setattr(self, cache, val)
        return val

    getter.__doc__ = prop.__doc__

    return property(getter)


class UserItemMapDataset(MapDataset):
    def __init__(
        self,
        user_item_df,
        user_col="user",
        item_col="id",
        sort_col="timestamp",
        max_window_size_lr=10,
        max_sequence_length=20,
        session_col=None,
    ):
        super().__init__()

        # populate anchors and targets
        self.anchors, self.targets = UserItemMapDataset.to_anchors_targets(
            user_item_df,
            user_col=user_col,
            item_col=item_col,
            sort_col=sort_col,
            max_window_size_lr=max_window_size_lr,
            max_sequence_length=max_sequence_length,
        )

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        return self.anchors[index], self.targets[index]

    @staticmethod
    def get_target_items(sequence, anchor_index, window_size=2):
        rand_num_items_lr = np.random.randint(1, window_size + 1)
        start = (
            anchor_index - rand_num_items_lr
            if (anchor_index - rand_num_items_lr) > 0
            else 0
        )
        stop = anchor_index + rand_num_items_lr
        target_items = (
            sequence[start:anchor_index] + sequence[anchor_index + 1 : stop + 1]
        )
        return list(target_items)

    @staticmethod
    def to_anchors_targets(
        user_item_df,
        user_col="user",
        item_col="id",
        sort_col="timestamp",
        max_window_size_lr=10,
        max_sequence_length=20,
        session_col=None,
    ):
        anchors, targets = list(), list()
        iter_upper_bound = max_sequence_length - max_window_size_lr
        groupbycols = [user_col, session_col] if session_col else [user_col]
        for user_id, user_df in user_item_df.sort_values(
            [user_col, sort_col], ascending=True
        ).groupby(groupbycols):
            id_sequence = user_df[item_col].tolist()
            id_sequence.reverse()  # most recent first
            id_sequence = (
                id_sequence[:max_sequence_length]
                if len(id_sequence) > max_sequence_length
                else id_sequence
            )
            for anchor_index in range(0, min(iter_upper_bound, len(id_sequence))):
                _targets = UserItemMapDataset.get_target_items(
                    id_sequence, anchor_index, window_size=max_window_size_lr
                )  # stochastic method
                _anchors = [id_sequence[anchor_index]] * len(_targets)
                anchors += _anchors
                targets += _targets
        return anchors, targets


class UserItemIterableDataset(IterableDataset):
    def __init__(
        self,
        user_item_df,
        user_col="user",
        item_col="id",
        sort_col="timestamp",
        max_window_size_lr=10,
        max_sequence_length=20,
        session_timedelta="800s",
        shuffle=True,
    ):
        super().__init__()
        self.df = user_item_df
        self.user_col = user_col
        self.item_col = item_col
        self.sort_col = sort_col
        self.max_sequence_length = max_sequence_length
        self.max_window_size_lr = max_window_size_lr
        self.shuffle = shuffle
        self.dataset = None

        # session identification
        self.sessions(timedelta=session_timedelta)

    def __iter__(self):
        self.dataset = UserItemMapDataset(
            self.df,
            user_col=self.user_col,
            item_col=self.item_col,
            sort_col=self.sort_col,
            max_window_size_lr=self.max_window_size_lr,
            max_sequence_length=self.max_sequence_length,
            session_col="session_nbr",
        )
        if self.shuffle:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)
        logger.debug(f"built stochastic skip-gram dataset n=({len(self.dataset):,})")
        for rand_index in sampler:
            yield self.dataset[rand_index]

    def sessions(self, timedelta="800s"):
        self.df["session_end"] = (
            self.df.sort_values([self.user_col, self.sort_col], ascending=True)
            .groupby(self.user_col)[self.sort_col]
            .diff(periods=1)
            > pd.Timedelta(timedelta)
        ).astype(int)
        self.df["session_nbr"] = self.df.groupby(self.user_col)["session_end"].cumsum() + 1
        self.df["session_id"] = self.df[self.user_col].astype(str) + "-" + self.df["session_nbr"].astype(str)

    @staticmethod
    def item_frequencies(df, item_col="id"):
        return df[item_col].value_counts(normalize=True).sort_index()

    @staticmethod
    def subsample(df, item_col="id", thresh=1e-5):
        item_freq = UserItemIterableDataset.item_frequencies(df, item_col)
        discard_dist = 1 - np.sqrt(thresh / item_freq)
        subsampled = discard_dist.loc[
            (1 - discard_dist) > np.random.random(size=len(discard_dist))
        ]
        return subsampled.index.tolist()

    @staticmethod
    def item_distribution(df, item_col="id", p=0.75):
        item_freq = UserItemIterableDataset.item_frequencies(df, item_col)
        item_dist = (item_freq ** (p)) / np.sum(item_freq ** (p))
        return item_dist
