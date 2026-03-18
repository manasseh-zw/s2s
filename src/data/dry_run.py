import modal
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("shona-data-dry-run")

# Updated Cloud Environment Definition
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")  # CRITICAL: Linux system-level audio decoder
    .pip_install(
        "datasets[audio]",  # Tells HF to install all audio dependencies
        "huggingface_hub", 
        "soundfile", 
        "librosa", 
        "numpy", 
        "python-dotenv", 
        "webrtcvad",
        "torchcodec",       # The specific library it crashed asking for
        "torchaudio"
    )
)

@app.function(
    image=image, 
    cpu=8.0, 
    memory=16384, 
    timeout=3600,
    secrets=[modal.Secret.from_dotenv()]
)
def process_dry_run(sample_size=1000, top_k=20):
    import librosa
    import webrtcvad
    from datasets import load_dataset
    from huggingface_hub import login
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env")
    
    login(token=hf_token)

    print(f"📡 Streaming {sample_size} Shona samples from Google WaxalNLP...")
    
    # streaming=True bypasses the 100GB download. 
    # It only pulls Shona data as we iterate.
    stream_dataset = load_dataset(
        "google/WaxalNLP", 
        "sna_asr", 
        split="train", 
        streaming=True
    )
    
    # Pull exactly the number of samples we want to test
    subset = stream_dataset.take(sample_size)
    
    # Convert stream to list so we can use .map-like logic and sorting
    raw_data = list(subset)
    print(f"📥 Successfully pulled {len(raw_data)} samples into memory.")

    # --- LAYER 1: Length Filter ---
    # We apply this to the list we just pulled
    filtered_data = [
        x for x in raw_data 
        if 5.0 <= (len(x["audio"]["array"]) / x["audio"]["sampling_rate"]) <= 22.0
    ]
    print(f"🧹 {len(filtered_data)} samples remaining after length filter.")

    # --- LAYER 2: Quality Scoring (WebRTC VAD) ---
    def calculate_quality(item):
        TARGET_SR = 16000
        VAD_MODE = 3
        INVALID_SCORE = -999.0
        EPSILON = 1e-10
        frame_ms = 30
        frame_len = int(TARGET_SR * frame_ms / 1000)

        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]

        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        
        num_f = len(audio) // frame_len
        if num_f == 0: return INVALID_SCORE, 0.0, 0.0
        
        audio = audio[:num_f * frame_len]
        frames = audio.reshape(-1, frame_len)
        powers = np.mean(frames ** 2, axis=1)

        pcm = (audio * 32767).astype(np.int16).reshape(-1, frame_len)
        vad = webrtcvad.Vad(VAD_MODE)
        
        mask = np.array([vad.is_speech(f.tobytes(), TARGET_SR) for f in pcm], dtype=bool)
        ratio = np.mean(mask)
        
        if ratio == 0: return INVALID_SCORE, 0.0, 0.0

        s_p = np.mean(powers[mask])
        n_p_list = powers[~mask]
        
        if n_p_list.size >= 3:
            q_count = max(1, int(np.ceil(n_p_list.size * 0.2)))
            n_p = np.mean(np.partition(n_p_list, q_count - 1)[:q_count])
            n_src = "non_speech"
        else:
            q_count = max(1, int(np.ceil(len(powers) * 0.1)))
            n_p = np.mean(np.partition(powers, q_count - 1)[:q_count])
            n_src = "floor"

        snr = float(10 * np.log10((s_p + EPSILON) / (n_p + EPSILON)))
        penalty = 0.0
        if ratio < 0.35: penalty += (0.35 - ratio) * 20.0
        if ratio > 0.95: penalty += (ratio - 0.95) * 40.0
        if n_src == "floor": penalty += 3.0
        
        return snr - penalty, snr, ratio

    print("🧮 Calculating VAD Quality Scores...")
    scored_data = []
    for item in filtered_data:
        q_score, snr, ratio = calculate_quality(item)
        if q_score > -999.0:
            item["quality_score"] = q_score
            item["snr"] = snr
            item["ratio"] = ratio
            scored_data.append(item)

    # --- LAYER 3: Rank and Prepare ---
    scored_data.sort(key=lambda x: x["quality_score"], reverse=True)
    top_dataset = scored_data[:top_k]

    results = []
    for item in top_dataset:
        results.append({
            "audio": item["audio"]["array"],
            "sr": item["audio"]["sampling_rate"],
            "quality": item["quality_score"],
            "snr": item["snr"],
            "ratio": item["ratio"],
            "transcription": item["transcription"]
        })
    return results

@app.local_entrypoint()
def main():
    import soundfile as sf
    print("🛠️ Initiating Surgical Stream Dry Run...")
    
    clips = process_dry_run.remote(sample_size=1000, top_k=20)
    
    output_dir = "test_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n📥 Received {len(clips)} samples. Saving to local '{output_dir}' folder...")
    
    for i, clip in enumerate(clips):
        clean_text = "".join(x for x in clip['transcription'][:20] if x.isalnum())
        filename = f"{output_dir}/rank_{i+1:02d}_score_{clip['quality']:.1f}_{clean_text}.wav"
        sf.write(filename, clip['audio'], clip['sr'])
        print(f"✅ Saved: {filename} (SNR: {clip['snr']:.1f})")
        
    print(f"\n🎧 DONE! Open 'src/{output_dir}' and listen to the results.")