from collections.abc import Iterator, Sequence
import hashlib
import logging
import multiprocessing
import os
import re
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.rlds_pi05_dataset import Pi05RldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class InjectEpisodeLenFromEpisodeIndex(_transforms.DataTransformFn):
    def __init__(self, episode_lengths: np.ndarray):
        self._episode_lengths = np.asarray(episode_lengths, dtype=np.int32)

    def __call__(self, data: dict) -> dict:
        if "episode_len" in data or "episode_index" not in data:
            return data

        episode_index = int(np.asarray(data["episode_index"]).item())
        if episode_index < 0 or episode_index >= self._episode_lengths.shape[0]:
            return data

        data = dict(data)
        data["episode_len"] = np.asarray(self._episode_lengths[episode_index], dtype=np.int32)
        return data


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _drop_non_numeric_fields(batch: dict) -> dict:
    batch = dict(batch)
    for key in list(batch.keys()):
        if isinstance(batch[key], dict):
            continue
        dtype = np.asarray(batch[key]).dtype
        if dtype.kind in {"O", "S", "U"}:
            batch.pop(key)
    return batch


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        # TorchCodec wheels can be ABI-sensitive on some clusters. Default to
        # pyav for robust decoding; override with OPENPI_LEROBOT_VIDEO_BACKEND.
        video_backend=os.environ.get("OPENPI_LEROBOT_VIDEO_BACKEND", "pyav"),
    )

    if hasattr(dataset, "episode_data_index"):
        episode_data_index = dataset.episode_data_index
        if isinstance(episode_data_index, dict) and "from" in episode_data_index and "to" in episode_data_index:
            episode_from = np.asarray(episode_data_index["from"], dtype=np.int64)
            episode_to = np.asarray(episode_data_index["to"], dtype=np.int64)
            episode_lengths = np.maximum(episode_to - episode_from, 1).astype(np.int32)
            dataset = TransformedDataset(dataset, [InjectEpisodeLenFromEpisodeIndex(episode_lengths)])

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    if data_config.adapter_kind is not None:
        return Pi05RldsDataset(
            data_dir=data_config.rlds_data_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            action_chunk_size=action_horizon,
            adapter_kind=data_config.adapter_kind,
            datasets=data_config.datasets,
        )

    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, at.Array, at.Array]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")
    require_accurate_progress = bool(config.enable_progress_loss and getattr(config.model, "enable_progress_head", False))
    progress_target_mode = config.progress_target_mode
    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
            require_accurate_progress=require_accurate_progress,
            progress_target_mode=progress_target_mode,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        require_accurate_progress=require_accurate_progress,
        progress_target_mode=progress_target_mode,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    require_accurate_progress: bool = False,
    progress_target_mode: Literal["chunk"] = "chunk",
) -> DataLoader[tuple[_model.Observation, _model.Actions, at.Array, at.Array]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        require_accurate_progress=require_accurate_progress,
        progress_target_mode=progress_target_mode,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
    require_accurate_progress: bool = False,
    progress_target_mode: Literal["chunk"] = "chunk",
) -> DataLoader[tuple[_model.Observation, _model.Actions, at.Array, at.Array]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
        require_accurate_progress=require_accurate_progress,
        progress_target_mode=progress_target_mode,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        require_accurate_progress: bool = False,
        progress_target_mode: Literal["chunk"] = "chunk",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._require_accurate_progress = require_accurate_progress
        self._progress_target_mode = progress_target_mode

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                batch = _pad_actions_to_episode_end(batch, progress_target_mode=self._progress_target_mode)
                progress_target, episode_hash = _compute_progress_and_episode_hash(
                    batch,
                    require_accurate_progress=self._require_accurate_progress,
                    progress_target_mode=self._progress_target_mode,
                )
                batch["progress_target"] = progress_target
                batch["episode_hash"] = episode_hash
                batch = _drop_non_numeric_fields(batch)
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        require_accurate_progress: bool = False,
        progress_target_mode: Literal["chunk"] = "chunk",
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches
        self._require_accurate_progress = require_accurate_progress
        self._progress_target_mode = progress_target_mode

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                batch = _pad_actions_to_episode_end(batch, progress_target_mode=self._progress_target_mode)
                progress_target, episode_hash = _compute_progress_and_episode_hash(
                    batch,
                    require_accurate_progress=self._require_accurate_progress,
                    progress_target_mode=self._progress_target_mode,
                )
                batch["progress_target"] = progress_target
                batch["episode_hash"] = episode_hash
                batch = _drop_non_numeric_fields(batch)
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            progress_target = batch.pop("progress_target")
            episode_hash = batch.pop("episode_hash")
            yield _model.Observation.from_dict(batch), batch["actions"], progress_target, episode_hash


_STEP_ID_SUFFIX_RE = re.compile(r"^(.*)--(\d+)$")
def _stable_hash_int64(text: str) -> np.int64:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.int64(int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1))


def _decode_text_array(arr: np.ndarray) -> list[str]:
    flat = arr.reshape(-1)
    out = []
    for value in flat:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _compute_progress_and_episode_hash(
    batch: dict[str, np.ndarray],
    *,
    require_accurate_progress: bool = False,
    progress_target_mode: Literal["chunk"] = "chunk",
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(batch["actions"])
    batch_size = actions.shape[0]
    action_horizon = actions.shape[1]

    def _compute_chunk_progress_from_frame_index(frame_index: np.ndarray, episode_len: np.ndarray) -> np.ndarray:
        offsets = np.arange(action_horizon, dtype=np.float32)[None, :]
        denom = np.maximum(episode_len - 1.0, 1.0)[:, None]
        progress_base = (frame_index[:, None] + offsets) / denom
        return np.clip(progress_base.astype(np.float32), 0.0, 1.0)

    def _decode_step_metadata() -> tuple[np.ndarray, list[str]]:
        step_ids = _decode_text_array(np.asarray(batch["step_id"]))
        frame_index = np.zeros(batch_size, dtype=np.float32)
        episode_keys: list[str] = []
        for i, step_id in enumerate(step_ids):
            match = _STEP_ID_SUFFIX_RE.match(step_id)
            if match is None:
                episode_keys.append(step_id)
                frame_index[i] = 0.0
            else:
                episode_keys.append(match.group(1))
                frame_index[i] = float(match.group(2))
        return frame_index, episode_keys

    if "frame_index" in batch and "episode_len" in batch:
        frame_index = np.asarray(batch["frame_index"], dtype=np.float32).reshape(batch_size)
        episode_len = np.asarray(batch["episode_len"], dtype=np.float32).reshape(batch_size)
        progress = _compute_chunk_progress_from_frame_index(frame_index, episode_len)
    else:
        if require_accurate_progress:
            keys = ", ".join(sorted(batch.keys()))
            raise ValueError(
                "Accurate progress supervision requires linear trajectory progress from "
                "(`frame_index`, `episode_len`). "
                f"Available batch keys: {keys}"
            )
        progress = np.zeros((batch_size, action_horizon), dtype=np.float32)

    progress = np.clip(progress.astype(np.float32), 0.0, 1.0)

    if "episode_hash" in batch:
        episode_hash = np.asarray(batch["episode_hash"], dtype=np.int64).reshape(batch_size)
    elif "step_id" in batch:
        _, episode_keys = _decode_step_metadata()
        episode_hash = np.asarray([_stable_hash_int64(k) for k in episode_keys], dtype=np.int64)
    elif "episode_index" in batch:
        episode_hash = np.asarray(batch["episode_index"], dtype=np.int64).reshape(batch_size)
    else:
        if "prompt" in batch:
            prompts = _decode_text_array(np.asarray(batch["prompt"]))
            episode_hash = np.asarray([_stable_hash_int64(p) for p in prompts], dtype=np.int64)
        else:
            episode_hash = np.asarray([_stable_hash_int64(f"batch_local_{i}") for i in range(batch_size)], dtype=np.int64)

    return progress, episode_hash


def _pad_actions_to_episode_end(
    batch: dict[str, np.ndarray], *, progress_target_mode: Literal["chunk"] = "chunk"
) -> dict[str, np.ndarray]:
    if "frame_index" not in batch or "episode_len" not in batch:
        return batch

    actions = np.asarray(batch["actions"])
    if actions.ndim < 3 or actions.shape[1] == 0:
        return batch

    frame_index = np.asarray(batch["frame_index"], dtype=np.float32).reshape(actions.shape[0])
    episode_len = np.asarray(batch["episode_len"], dtype=np.float32).reshape(actions.shape[0])
    last_valid_offset = np.clip((episode_len - 1.0 - frame_index).astype(np.int32), 0, actions.shape[1] - 1)
    padded_actions = np.array(actions, copy=True)

    for i, last_offset in enumerate(last_valid_offset):
        if last_offset + 1 < padded_actions.shape[1]:
            padded_actions[i, last_offset + 1 :] = padded_actions[i, last_offset]

    batch = dict(batch)
    batch["actions"] = padded_actions
    return batch
