import modal
import os

app = modal.App("sna-data-patcher")
refined_vol = modal.Volume.from_name("sna-refined-vol")


@app.function(
    image=modal.Image.debian_slim().pip_install(
        "datasets[audio]",
        "huggingface_hub",
        "soundfile",
    ),
    volumes={"/refined": refined_vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=10800,
)
def patch_and_push():
    from datasets import load_from_disk, Audio, DatasetDict
    from huggingface_hub import login
    import soundfile as sf

    def make_audio_writer(split_audio_dir):
        def materialize_audio(example, idx):
            audio = example["audio"]
            audio_path = os.path.join(split_audio_dir, f"{idx:05d}.wav")
            sf.write(
                audio_path,
                audio["array"],
                audio["sampling_rate"],
                format="WAV",
                subtype="FLOAT",
            )
            example["audio"] = audio_path
            return example

        return materialize_audio

    hf_token = os.environ.get("HF_TOKEN")
    hf_user = os.environ.get("HF_USERNAME")
    if not hf_token or not hf_user:
        raise RuntimeError("Missing HF_TOKEN or HF_USERNAME")

    login(token=hf_token)
    repo_id = f"{hf_user}/sna-tts-refined-v2"

    print("📝 Loading V2 dataset from volume...")
    ds = load_from_disk("/refined/sna_refined_v2")

    patched_audio_dir = "/refined/sna_refined_v2_audio_files"
    patched_splits = {}

    for split_name, split_ds in ds.items():
        split_audio_dir = os.path.join(patched_audio_dir, split_name)
        os.makedirs(split_audio_dir, exist_ok=True)

        print(f"🎧 Materializing '{split_name}' audio files...")
        patched_split = split_ds.map(make_audio_writer(split_audio_dir), with_indices=True)
        patched_split = patched_split.cast_column("audio", Audio())
        patched_splits[split_name] = patched_split

    ds = DatasetDict(patched_splits)

    print("💾 Saving patched version to volume...")
    ds.save_to_disk("/refined/sna_refined_v2_patched")
    refined_vol.commit()

    print(f"🚀 Pushing metadata fix to Hugging Face: {repo_id}...")
    ds.push_to_hub(repo_id)
    print("✅ All done! Go check the HF viewer now.")


if __name__ == "__main__":
    patch_and_push.remote()
