import modal

app = modal.App("sna-data-ingestor")
raw_vol = modal.Volume.from_name("sna-raw-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("datasets[audio]", "huggingface_hub", "torchcodec")
)

@app.function(
    image=image,
    cpu=4.0, 
    memory=32768, 
    timeout=3600,
    volumes={"/raw_data": raw_vol},
    secrets=[modal.Secret.from_dotenv()]
)
def ingest_shona_split():
    from datasets import load_dataset

    print("🚀 Downloading 'sna_asr' labeled train split directly from Hugging Face...")
    dataset = load_dataset("google/WaxalNLP", "sna_asr", split="train")

    print(f"📥 Successfully loaded {len(dataset)} records. Saving to volume...")
    dataset.save_to_disk("/raw_data/sna_raw_v1")

    raw_vol.commit()
    print("✅ Ingestion complete. Raw data safely stored at /raw_data/sna_raw_v1")

if __name__ == "__main__":
    ingest_shona_split.remote()