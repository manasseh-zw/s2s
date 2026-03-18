import modal
import os

app = modal.App("sna-data-uploader")
refined_vol = modal.Volume.from_name("sna-refined-vol")

@app.function(
    image=modal.Image.debian_slim().pip_install("datasets", "huggingface_hub"),
    volumes={"/refined": refined_vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=10800 # 3 hours
)
def upload_dataset():
    from datasets import load_from_disk
    from huggingface_hub import login
    
    # 1. Strict Token Check
    hf_token = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME")
    
    if not hf_token or not hf_username:
        raise ValueError("🚨 Missing HF_TOKEN or HF_USERNAME in your .env file!")
        
    # Authenticate with HF
    login(token=hf_token)
    
    repo_id = f"{hf_username}/sna-tts-refined-v2"
    
    print("🚀 Loading V2 (Normalized) dataset from volume...")
    ds = load_from_disk("/refined/sna_refined_v2")
    
    print(f"☁️ Pushing to Hugging Face: {repo_id}...")
    ds.push_to_hub(repo_id)
    print("✅ Upload complete! The data is now safe in the cloud.")

if __name__ == "__main__":
    upload_dataset.remote()