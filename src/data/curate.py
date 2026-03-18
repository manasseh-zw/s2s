import modal
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("shona-data-refinery")

# Added webrtcvad to the pip installs
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("datasets", "huggingface_hub", "soundfile", "librosa", "numpy", "python-dotenv", "webrtcvad")
)

@app.function(
    image=image, 
    cpu=8.0, 
    memory=16384, 
    timeout=7200,
    secrets=[modal.Secret.from_dotenv()]
)
def curate_and_push_dataset():
    import numpy as np
    import librosa
    import webrtcvad
    from datasets import load_dataset
    from huggingface_hub import login
    
    hf_token = os.environ.get("HF_TOKEN")
    hf_user = os.environ.get("HF_USERNAME")
    
    if not hf_token or not hf_user:
        logger.error("❌ Missing environment variables! Check HF_TOKEN and HF_USERNAME in .env")
        return

    logger.info(f"✅ HF_TOKEN detected (starts with: {hf_token[:4]}...)")
    login(token=hf_token)

    print("🚀 Downloading Google's WaxalNLP Shona dataset (train split)...")
    dataset = load_dataset("google/WaxalNLP", "sna_asr", split="train")

    # ==========================================
    # LAYER 1: The Length Filter (5.0s - 22.0s)
    # ==========================================
    dataset = dataset.filter(
        lambda x: 5.0 <= (len(x["audio"]["array"]) / x["audio"]["sampling_rate"]) <= 22.0, 
        num_proc=8
    )
    print(f"🧹 Rows after length filter: {len(dataset)}")

    # ==========================================
    # LAYER 2: The WebRTC VAD Quality Scoring
    # ==========================================
    def calculate_quality(example):
        TARGET_SR = 16000 # WebRTC strictly requires 16kHz or 32kHz
        VAD_MODE = 3      # 3 is the most aggressive filter for weeding out noise
        INVALID_SCORE = -999.0
        EPSILON = 1e-10
        
        # WebRTC requires exact 10, 20, or 30ms frames. We use 30ms.
        frame_duration_ms = 30
        frame_length = int(TARGET_SR * frame_duration_ms / 1000) # 480 samples

        def invalid_clip(reason):
            return {
                "snr_score": INVALID_SCORE, "quality_score": INVALID_SCORE,
                "speech_seconds": 0.0, "speech_ratio": 0.0, "noise_source": reason
            }

        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sample_rate = example["audio"]["sampling_rate"]

        # Resample if necessary for WebRTC
        if sample_rate != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR

        # Truncate audio so it perfectly divides into 30ms frames (prevents reshape crashes)
        num_frames = len(audio) // frame_length
        if num_frames == 0: return invalid_clip("too_short")
        audio = audio[:num_frames * frame_length]

        frames = audio.reshape(-1, frame_length)
        frame_powers = np.mean(frames ** 2, axis=1)

        # Convert to 16-bit PCM for WebRTC
        pcm_audio = (audio * 32767).astype(np.int16).reshape(-1, frame_length)
        vad = webrtcvad.Vad(VAD_MODE)
        
        speech_mask = np.array(
            [vad.is_speech(frame.tobytes(), sample_rate) for frame in pcm_audio],
            dtype=bool,
        )

        speech_seconds = np.sum(speech_mask) * (frame_duration_ms / 1000.0)
        speech_ratio = np.mean(speech_mask)
        
        if speech_ratio == 0: return invalid_clip("no_speech")

        speech_powers = frame_powers[speech_mask]
        non_speech_powers = frame_powers[~speech_mask]

        # Estimate noise floor using the quietest percentiles
        if non_speech_powers.size >= 3:
            quiet_count = max(1, int(np.ceil(non_speech_powers.size * 0.2)))
            noise_candidates = np.partition(non_speech_powers, quiet_count - 1)[:quiet_count]
            noise_source = "non_speech_floor"
        else:
            # Fallback for tightly trimmed clips
            quiet_count = max(1, int(np.ceil(len(frame_powers) * 0.1)))
            noise_candidates = np.partition(frame_powers, quiet_count - 1)[:quiet_count]
            noise_source = "clip_floor"

        signal_power = float(np.mean(speech_powers))
        noise_power = float(np.mean(noise_candidates))
        snr_score = float(10 * np.log10((signal_power + EPSILON) / (noise_power + EPSILON)))

        # Penalize clips that are badly trimmed
        reliability_penalty = 0.0
        if speech_ratio < 0.35: reliability_penalty += (0.35 - speech_ratio) * 20.0
        if speech_ratio > 0.95: reliability_penalty += (speech_ratio - 0.95) * 40.0
        if noise_source == "clip_floor": reliability_penalty += 3.0

        quality_score = snr_score - reliability_penalty
        
        return {
            "snr_score": snr_score,
            "quality_score": quality_score,
            "speech_seconds": speech_seconds,
            "speech_ratio": speech_ratio,
            "noise_source": noise_source,
        }

    print("🧮 Layer 2: Running WebRTC VAD Quality Analysis (Cloud Parallel)...")
    dataset = dataset.map(calculate_quality, num_proc=8)
    
    print("🗑️ Removing invalid/silent clips...")
    dataset = dataset.filter(lambda x: x["quality_score"] > -999.0, num_proc=8)

    # ==========================================
    # LAYER 3: Rank and Publish
    # ==========================================
    print("🏆 Layer 3: Selecting top 5,000 elite clips...")
    dataset = dataset.sort("quality_score", reverse=True)
    final_set = dataset.select(range(min(5000, len(dataset))))

    repo_id = f"{hf_user}/shona-tts-dataset-curated"
    print(f"☁️ Pushing final Gold Standard dataset to https://huggingface.co/datasets/{repo_id}...")
    final_set.push_to_hub(repo_id, private=False)
    print("✅ Process complete!")

if __name__ == "__main__":
    curate_and_push_dataset.remote()