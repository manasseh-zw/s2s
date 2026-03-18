"""
Modal orchestrator for the owned Shona CSM training pipeline.

This runner keeps the official CSM repository external while executing the
project-owned pretokenization and training loop from `src/models`.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "sna-csm-trainer"
CSM_COMMIT = "836f886515f0dec02c22ed2316cc78904bdc0f36"
LOCAL_MODELS_DIR = Path(__file__).resolve().parent
REMOTE_MODELS_DIR = Path("/workspace/project/src/models")

app = modal.App(APP_NAME)
refined_vol = modal.Volume.from_name("sna-refined-vol")
model_vol = modal.Volume.from_name("sna-model-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .add_local_dir(str(LOCAL_MODELS_DIR), remote_path=str(REMOTE_MODELS_DIR))
    .pip_install(
        "datasets[audio]>=4.8.2",
        "h5py>=3.10.0",
        "huggingface_hub==0.28.1",
        "moshi==0.2.2",
        "numpy>=2.0.0",
        "python-dotenv>=1.0.1",
        "PyYAML>=6.0.2",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "setuptools>=70.0.0",
        "tokenizers==0.21.0",
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "torchao==0.9.0",
        "torchtune==0.4.0",
        "tqdm>=4.66.0",
        "transformers==4.49.0",
        "wandb>=0.19.6",
        "wheel>=0.43.0",
        "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master",
    )
    .run_commands("git clone https://github.com/SesameAILabs/csm.git /workspace/csm")
    .run_commands(f"cd /workspace/csm && git checkout {CSM_COMMIT}")
)


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    volumes={"/refined": refined_vol, "/checkpoints": model_vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=86400,
)
def run_training(
    dataset_path: str = "/refined/sna_refined_v2",
    token_path: str = "/checkpoints/tokens/sna_refined_v2.hdf5",
    output_dir: str = "/checkpoints/runs/sna_csm_run1",
    overwrite_tokens: bool = True,
) -> None:
    os.environ["CSM_REPO_PATH"] = "/workspace/csm"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config_path = REMOTE_MODELS_DIR / "configs" / "shona_finetune.yaml"
    pretokenize_script = REMOTE_MODELS_DIR / "pretokenize.py"
    train_script = REMOTE_MODELS_DIR / "train.py"
    token_audit_path = output_dir_path / "audit" / "pretokenize_summary.json"
    orchestrator_path = output_dir_path / "audit" / "orchestrator.json"

    orchestrator_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator_path.write_text(
        json.dumps(
            {
                "dataset_path": dataset_path,
                "token_path": token_path,
                "output_dir": output_dir,
                "overwrite_tokens": overwrite_tokens,
                "config_path": str(config_path),
                "csm_commit": CSM_COMMIT,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    pretokenize_args = [
        "python3",
        str(pretokenize_script),
        "--dataset_path",
        dataset_path,
        "--output",
        token_path,
        "--audit_path",
        str(token_audit_path),
        "--device",
        "cuda",
    ]
    if overwrite_tokens:
        pretokenize_args.append("--overwrite")

    print("Starting direct HF-to-HDF5 pretokenization...")
    subprocess.run(pretokenize_args, check=True)
    model_vol.commit()

    print("Launching A100 training run...")
    subprocess.run(
        [
            "python3",
            str(train_script),
            "--data",
            token_path,
            "--output_dir",
            output_dir,
            "--config",
            str(config_path),
            "--n_epochs",
            "3",
            "--use_amp",
            "--wandb_project",
            "shona-csm-finetune",
            "--gen_sentences",
            "mhoro, unonzi ani?",
        ],
        check=True,
    )
    model_vol.commit()


if __name__ == "__main__":
    run_training.remote()
