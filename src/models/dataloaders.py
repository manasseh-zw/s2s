"""
Lean HDF5 dataloaders for the Shona CSM training pipeline.

Adapted from `knottwill/sesame-finetune`, keeping the bucketed batching logic
while trimming the implementation down to the train/val workflow we use.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

AUDIO_NUM_CODEBOOKS = int(os.getenv("AUDIO_NUM_CODEBOOKS", "32"))


class TokenizedDataset(Dataset):
    def __init__(self, token_dataset_path: str, split: str, load_in_memory: bool = False):
        if not token_dataset_path.endswith(".hdf5"):
            raise ValueError("Token dataset path must end with .hdf5")

        self.token_dataset_path = token_dataset_path
        self.split = split
        self.load_in_memory = load_in_memory
        self._file: h5py.File | None = None
        self._audio: list[torch.Tensor] | None = None
        self._text: list[torch.Tensor] | None = None

        with h5py.File(self.token_dataset_path, "r") as handle:
            self.length = len(handle[f"{self.split}/audio"])
            if self.load_in_memory:
                self._audio = [
                    torch.tensor(audio, dtype=torch.long)
                    for audio in handle[f"{self.split}/audio"][:]
                ]
                self._text = [
                    torch.tensor(text, dtype=torch.long)
                    for text in handle[f"{self.split}/text"][:]
                ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.load_in_memory:
            flat_audio = self._audio[index]
            text = self._text[index]
        else:
            if self._file is None:
                self._file = h5py.File(self.token_dataset_path, "r")
            flat_audio = torch.tensor(self._file[f"{self.split}/audio"][index], dtype=torch.long)
            text = torch.tensor(self._file[f"{self.split}/text"][index], dtype=torch.long)

        audio = flat_audio.view(AUDIO_NUM_CODEBOOKS, -1)
        return {"audio": audio, "text": text}

    def __del__(self) -> None:
        if self._file is not None:
            self._file.close()


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens: list[torch.Tensor] = []
    tokens_mask: list[torch.Tensor] = []

    for item in batch:
        audio_tokens = item["audio"]
        text_tokens = item["text"]

        eos_frame = torch.zeros((audio_tokens.size(0), 1), dtype=torch.long)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(
            (audio_tokens.size(1), AUDIO_NUM_CODEBOOKS + 1),
            dtype=torch.long,
        )
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)

        audio_frame_mask = torch.zeros_like(audio_frame, dtype=torch.bool)
        audio_frame_mask[:, :-1] = True

        text_frame = torch.zeros((len(text_tokens), AUDIO_NUM_CODEBOOKS + 1), dtype=torch.long)
        text_frame[:, -1] = text_tokens

        text_frame_mask = torch.zeros_like(text_frame, dtype=torch.bool)
        text_frame_mask[:, -1] = True

        tokens.append(torch.cat([text_frame, audio_frame], dim=0))
        tokens_mask.append(torch.cat([text_frame_mask, audio_frame_mask], dim=0))

    return (
        pad_sequence(tokens, batch_first=True),
        pad_sequence(tokens_mask, batch_first=True, padding_value=False),
    )


class BucketSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, random_seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.bins = self._create_bins()

    def _create_bins(self) -> list[list[int]]:
        sorted_pairs = sorted(enumerate(self.lengths), key=lambda item: item[1])
        bins: list[list[int]] = []
        current_bin: list[int] = []

        for index, _ in sorted_pairs:
            current_bin.append(index)
            if len(current_bin) == self.batch_size:
                bins.append(current_bin)
                current_bin = []

        if current_bin:
            bins.append(current_bin)

        return bins

    def __iter__(self):
        bins = [bin_indices[:] for bin_indices in self.bins]
        if self.shuffle:
            rng = np.random.RandomState(self.random_seed)
            rng.shuffle(bins)
            for bin_indices in bins:
                rng.shuffle(bin_indices)
        yield from bins

    def __len__(self) -> int:
        return len(self.bins)


def load_lengths(token_dataset_path: str, split: str) -> list[int]:
    with h5py.File(token_dataset_path, "r") as handle:
        return list(handle[f"{split}/length"][:])


def create_dataloaders(
    token_dataset_path: str,
    batch_size: int,
    load_in_memory: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_lengths = load_lengths(token_dataset_path, "train")
    val_lengths = load_lengths(token_dataset_path, "val")

    train_dataset = TokenizedDataset(
        token_dataset_path,
        split="train",
        load_in_memory=load_in_memory,
    )
    val_dataset = TokenizedDataset(
        token_dataset_path,
        split="val",
        load_in_memory=load_in_memory,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketSampler(train_lengths, batch_size=batch_size, shuffle=True),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=BucketSampler(val_lengths, batch_size=batch_size, shuffle=False),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
