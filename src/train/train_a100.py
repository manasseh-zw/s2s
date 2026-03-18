import modal
import os

app = modal.App("sna-csm-trainer")

refined_vol = modal.Volume.from_name("sna-refined-vol")
model_vol = modal.Volume.from_name("sna-model-vol", create_if_missing=True)

# 1. Build the Heavy Custom Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    # Clone Knottwill's finetuner
    .run_commands("git clone https://github.com/knottwill/sesame-finetune.git /workspace/sesame")
    # Clone the official CSM repo (Required by README)
    .run_commands("git clone https://github.com/SesameAILabs/csm.git /workspace/csm")
    .run_commands("cd /workspace/csm && git checkout 836f886515f0dec02c22ed2316cc78904bdc0f36")
    .workdir("/workspace/sesame")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("datasets", "soundfile", "wandb")
)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"), 
    volumes={"/refined": refined_vol, "/checkpoints": model_vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=86400 
)
def run_training():
    import subprocess
    import json
    import soundfile as sf
    from datasets import load_from_disk
    
    # --- SETUP ENVIRONMENT VARIABLES ---
    os.environ["CSM_REPO_PATH"] = "/workspace/csm"
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        subprocess.run(["wandb", "login", wandb_key], check=True)

    print("🚀 Loading V2 Dataset...")
    ds = load_from_disk("/refined/sna_refined_v2")
    
    # --- STEP 1: MATERIALIZE WAVS TO EPHEMERAL STORAGE ---
    # We write these to /tmp so they delete themselves after training and don't cost you storage!
    print("🎧 Exporting temporary WAV files for Pre-tokenization...")
    os.makedirs("/tmp/audio/train", exist_ok=True)
    os.makedirs("/tmp/audio/val", exist_ok=True)
    
    metadata = {"train": [], "val": []}
    
    # Map HF splits to Sesame splits (Sesame only uses train and val)
    split_mapping = {"train": "train", "valid": "val"} 
    
    for hf_split, sesame_split in split_mapping.items():
        for i, row in enumerate(ds[hf_split]):
            wav_path = f"/tmp/audio/{sesame_split}/{i:05d}.wav"
            # Write to disk
            sf.write(wav_path, row["audio"]["array"], 16000, subtype="FLOAT")
            # Append to Sesame metadata format
            metadata[sesame_split].append({
                "text": row["transcription"],
                "path": wav_path
            })
            
        # Save the JSON metadata
        with open(f"/tmp/{sesame_split}_metadata.json", "w") as f:
            json.dump(metadata[sesame_split], f)

    # --- STEP 2: PRE-TOKENIZATION ---
    print("🪄 Running pretokenize.py (Compressing to HDF5)...")
    hdf5_path = "/checkpoints/sna_tokens.hdf5"
    subprocess.run([
        "python", "pretokenize.py",
        "--train_data", "/tmp/train_metadata.json",
        "--val_data", "/tmp/val_metadata.json",
        "--output", hdf5_path,
        "--omit_speaker_id" # <--- The magic flag!
    ], check=True)
    
    model_vol.commit()

    # --- STEP 3: FINETUNING ---
    print("🔥 Starting A100 Training Loop (3 Epochs)...")
    train_cmd = [
        "python", "train.py",
        "--data", hdf5_path,
        "--config", "./configs/finetune_param_defaults.yaml",
        "--output_dir", "/checkpoints/sna_csm_run1",
        "--n_epochs", "3",
        "--use_amp",
        "--wandb_project", "shona-csm-finetune",
        "--gen_sentences", "mhoro, unonzi ani?", # Shona test sentence!
        "--gen_every", "500"
    ]
    
    subprocess.run(train_cmd, check=True)
    
    print("✅ Training complete! Syncing checkpoints to volume...")
    model_vol.commit()

if __name__ == "__main__":
    run_training.remote()