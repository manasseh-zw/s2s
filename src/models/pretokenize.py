"""
Direct Hugging Face to HDF5 pretokenization for the Shona CSM pipeline.

Adapted from `knottwill/sesame-finetune` and `SesameAILabs/csm`, but rewritten
for an in-memory `datasets` workflow with no temporary WAV materialization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from utils import AUDIO_NUM_CODEBOOKS, MIMI_SAMPLE_RATE, load_tokenizers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, default=Path("/refined/sna_refined_v2"))
    parser.add_argument("--output", type=Path, default=Path("/checkpoints/tokens/sna_refined_v2.hdf5"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def ensure_mono_audio(audio_array: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim == 0:
        return audio.reshape(1)
    if audio.ndim == 1:
        return np.nan_to_num(audio, copy=False)
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {audio.shape}")

    channel_axis = 0 if audio.shape[0] <= 8 and audio.shape[1] > 8 else 1
    audio = audio.mean(axis=channel_axis)
    return np.nan_to_num(audio, copy=False)


def prepare_waveform(audio_entry: dict, device: torch.device) -> torch.Tensor:
    waveform = ensure_mono_audio(audio_entry["array"])
    tensor = torch.from_numpy(waveform)
    sample_rate = int(audio_entry["sampling_rate"])

    if tensor.numel() == 0:
        raise ValueError("Encountered empty audio clip during pretokenization")

    if sample_rate != MIMI_SAMPLE_RATE:
        tensor = torchaudio.functional.resample(
            tensor,
            orig_freq=sample_rate,
            new_freq=MIMI_SAMPLE_RATE,
        )

    return tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def format_text(transcription: str) -> str:
    # Speaker IDs are intentionally omitted for this Shona-specific phonetic tune.
    return " ".join(str(transcription).split())


def append_to_hdf5(
    file_path: Path,
    split: str,
    audio_tokens_batch: list[np.ndarray],
    text_tokens_batch: list[np.ndarray],
) -> None:
    with h5py.File(file_path, "a") as handle:
        group = handle.require_group(split)
        vlen_dtype = h5py.special_dtype(vlen=np.int32)

        audio_ds = group.get("audio") or group.create_dataset(
            "audio",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_dtype,
        )
        text_ds = group.get("text") or group.create_dataset(
            "text",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_dtype,
        )
        length_ds = group.get("length") or group.create_dataset(
            "length",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
        )

        batch_size = len(audio_tokens_batch)
        audio_ds.resize(audio_ds.shape[0] + batch_size, axis=0)
        text_ds.resize(text_ds.shape[0] + batch_size, axis=0)
        length_ds.resize(length_ds.shape[0] + batch_size, axis=0)

        for index, (audio_tokens, text_tokens) in enumerate(
            zip(audio_tokens_batch, text_tokens_batch, strict=True)
        ):
            seq_len = audio_tokens.shape[0] // AUDIO_NUM_CODEBOOKS
            total_len = seq_len + len(text_tokens) + 1
            target_index = audio_ds.shape[0] - batch_size + index

            audio_ds[target_index] = audio_tokens
            text_ds[target_index] = text_tokens
            length_ds[target_index] = total_len


def resolve_validation_split(dataset: DatasetDict, preferred_name: str) -> str:
    if preferred_name in dataset:
        return preferred_name

    for fallback in ("valid", "validation", "val"):
        if fallback in dataset:
            return fallback

    raise KeyError("Could not find a validation split in the refined dataset")


def tokenize_split(
    dataset_split: Dataset,
    output_path: Path,
    output_split: str,
    audio_tokenizer,
    text_tokenizer,
    device: torch.device,
    save_every: int,
) -> None:
    audio_tokens_batch: list[np.ndarray] = []
    text_tokens_batch: list[np.ndarray] = []

    with torch.inference_mode():
        for example in tqdm(dataset_split, desc=f"Tokenizing {output_split}"):
            waveform = prepare_waveform(example["audio"], device)
            audio_tokens = audio_tokenizer.encode(waveform)[0]
            text_tokens = text_tokenizer.encode(format_text(example["transcription"]))

            audio_tokens_batch.append(
                audio_tokens.detach().cpu().numpy().astype(np.int32).reshape(-1)
            )
            text_tokens_batch.append(np.asarray(text_tokens, dtype=np.int32))

            if len(audio_tokens_batch) >= save_every:
                append_to_hdf5(output_path, output_split, audio_tokens_batch, text_tokens_batch)
                audio_tokens_batch.clear()
                text_tokens_batch.clear()

    if audio_tokens_batch:
        append_to_hdf5(output_path, output_split, audio_tokens_batch, text_tokens_batch)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Token file already exists at {args.output}. Pass --overwrite to rebuild it."
            )
        args.output.unlink()

    device = torch.device(args.device)
    dataset = load_from_disk(str(args.dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict saved to disk at the refined dataset path")

    valid_split_name = resolve_validation_split(dataset, args.valid_split)
    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    tokenize_split(
        dataset[args.train_split],
        output_path=args.output,
        output_split="train",
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        device=device,
        save_every=args.save_every,
    )
    tokenize_split(
        dataset[valid_split_name],
        output_path=args.output,
        output_split="val",
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        device=device,
        save_every=args.save_every,
    )

    print(f"Saved tokenized train/val splits to {args.output}")
    return args.output


if __name__ == "__main__":
    main()
