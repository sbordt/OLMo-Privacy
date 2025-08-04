#
# A modified version of IterableDataset that iterates over a subset of batches.
#
### BEGIN GAUSSIAN POISONING 
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.data

from ..aliases import PathOrStr
from ..torch_util import barrier, get_fs_local_rank, get_global_rank, get_world_size
from ..util import roundrobin, threaded_generator

__all__ = ["IterableDatasetSubset"]

log = logging.getLogger(__name__)


class IterableDatasetSubset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """
    Adapted from PyTorch's DistributedSampler, this wraps a Dataset or arbitrary sequence
    as an IterableDataset that can be deterministically restarted at any point by setting `start_index`,
    which should be a multiple of your global batch size.
    Similarly `max_examples`, if set, should be a multiple of global batch size.
    """

    def __init__(
        self,
        dataset: Union[Sequence[List[int]], Sequence[torch.Tensor], Sequence[Dict[str, Any]]],
        global_batch_size: int,
        subset_batch_indices: List[int] = [],
        *,
        seed: int = 0,
        epoch: int = 0,
        start_index: int = 0,
        max_examples: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        fs_local_rank: Optional[int] = None,
        work_dir: Optional[PathOrStr] = None,
        num_threads: Optional[int] = None,
    ):
        self.dataset = dataset
        self.subset_batch_indices = subset_batch_indices
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank if rank is not None else get_global_rank()
        self.fs_local_rank = fs_local_rank if fs_local_rank is not None else get_fs_local_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.total_size = global_batch_size * len(subset_batch_indices) * self.world_size # GAUSSIAN POISONING
        self.num_threads = num_threads
        assert global_batch_size % self.world_size == 0
        self.device_batch_size = global_batch_size // self.world_size
        self.global_indices_file: Optional[Path] = None
        self.work_dir = work_dir

        # GAUSSIAN POISONING: we assume that the global indices are already built and saved by the train_loader
        self.global_indices_file = Path(self.work_dir) / "global_indices.npy"
        # GAUSSIAN POISONING: we subset the dataset by subsetting the global indices
        # the global indices are a list of self.global_batch_size indices for every batch
        global_indices = np.memmap(self.global_indices_file, mode="r", dtype=np.uint32)  # type: ignore
        self.subset_indices = [global_indices[i : i + global_batch_size] for i in self.subset_batch_indices]
        self.subset_indices = np.array(self.subset_indices).flatten()
        

    def _build_and_save_global_indices(self):
        pass

    def get_global_indices(self) -> np.ndarray:
        return self.subset_indices

    def reshuffle(self, epoch: int):
        raise NotImplementedError(
            "Reshuffling is not implemented for IterableDatasetSubset. "
            "This dataset is designed to iterate over a fixed subset of indices."
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = self.get_global_indices()

        # Truncate to max_examples.
        if self.max_examples is not None:
            assert self.max_examples % self.world_size == 0
            indices = indices[: self.max_examples]

        # Start at the specified index.
        if self.start_index > 0:
            #  assert self.start_index % self.world_size == 0
            indices = indices[self.start_index :]

        # Slice indices by rank to avoid duplicates.
        indices = indices[self.rank : self.total_size : self.world_size]

        # Separate from data loading workers (which use multiprocessing), we also have the option
        # to use multi-threading (within workers).
        num_threads = self.num_threads

        # Slice the indices by data loader worker rank to avoid duplicates.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we should give worker 0 the first chunk of `device_batch_size` indices,
            # worker 1 the 2nd chunk of `device_train_batch_size` indices, etc...
            truncated_size = self.device_batch_size * (len(indices) // self.device_batch_size)
            left_overs = indices[truncated_size + worker_info.id :: worker_info.num_workers]
            indices = (
                indices[:truncated_size]
                .reshape((-1, self.device_batch_size))[worker_info.id :: worker_info.num_workers]  # type: ignore
                .reshape((-1,))
            )
            indices = np.concatenate([indices, left_overs])
        elif num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll try to guess
            # a good number of threads.
            num_threads = 4

        # Finally, potentially slice by threads.
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the batch size.
            queue_size = math.ceil(self.device_batch_size * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                generator = (self._get_dataset_item(int(idx)) for idx in indices[i::num_threads])
                thread_generators.append(
                    threaded_generator(generator, maxsize=queue_size, thread_name=f"data thread {i}")
                )

            return (x for x in roundrobin(*thread_generators))
        else:
            return (self._get_dataset_item(int(idx)) for idx in indices)

    def _get_dataset_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            return dict(**item, index=idx)
        elif dataclasses.is_dataclass(item):
            return dict(**dataclasses.asdict(item), index=idx)  # type: ignore
        else:
            return {"input_ids": item, "index": idx}
### END GAUSSIAN POISONING 