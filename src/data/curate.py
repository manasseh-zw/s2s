import modal
import os
import numpy as np
import heapq

app = modal.App("sna-data-refinery")
raw_vol = modal.Volume.from_name("sna-raw-vol")
refined_vol = modal.Volume.from_name("sna-refined-vol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("datasets[audio]", "librosa", "webrtcvad", "soundfile", "torchcodec")
)

@app.function(
    image=image,
    cpu=8.0, 
    memory=16384,
    timeout=7200,
    volumes={"/raw": raw_vol, "/refined": refined_vol},
    secrets=[modal.Secret.from_dotenv()]
)
def refine_shona_dataset():
    from datasets import load_from_disk, Dataset, DatasetDict
    import webrtcvad
    import librosa

    print("🚀 Loading raw records from sna-raw-vol...")
    dataset = load_from_disk("/raw/sna_raw_v1")

    vad = webrtcvad.Vad(2)
    TARGET_SR = 16000
    FRAME_MS = 30
    FRAME_LEN = int(TARGET_SR * FRAME_MS / 1000)

    def smooth_vad_mask(mask, min_speech_frames=3, bridge_gap_frames=2):
        mask = mask.copy()
        if mask.size == 0:
            return mask

        start = None
        runs = []
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))

        for run_start, run_end in runs:
            if (run_end - run_start) < min_speech_frames:
                mask[run_start:run_end] = False

        idx = np.where(mask)[0]
        if idx.size == 0:
            return mask

        for i in range(len(idx) - 1):
            gap = idx[i + 1] - idx[i] - 1
            if 0 < gap <= bridge_gap_frames:
                mask[idx[i] + 1:idx[i + 1]] = True

        return mask

    def score_clip(audio_16k, speech_mask):
        frames = audio_16k.reshape(-1, FRAME_LEN)
        frame_powers = np.mean(frames.astype(np.float32) ** 2, axis=1)

        speech_powers = frame_powers[speech_mask]
        non_speech_powers = frame_powers[~speech_mask]

        if speech_powers.size == 0:
            return None

        speech_seconds = speech_powers.size * (FRAME_MS / 1000.0)
        speech_ratio = speech_powers.size / len(frame_powers)

        if speech_seconds < 1.0:
            return None

        if non_speech_powers.size >= 4:
            quiet_n = max(1, int(np.ceil(non_speech_powers.size * 0.2)))
            noise_floor = np.mean(np.partition(non_speech_powers, quiet_n - 1)[:quiet_n])
            reliability_penalty = 0.0
        else:
            quiet_n = max(1, int(np.ceil(len(frame_powers) * 0.1)))
            noise_floor = np.mean(np.partition(frame_powers, quiet_n - 1)[:quiet_n])
            reliability_penalty = 3.0

        signal_power = np.mean(speech_powers)
        snr_db = 10.0 * np.log10((signal_power + 1e-10) / (noise_floor + 1e-10))

        if speech_ratio < 0.35:
            reliability_penalty += (0.35 - speech_ratio) * 20.0
        if speech_ratio > 0.95:
            reliability_penalty += (speech_ratio - 0.95) * 40.0

        quality_score = snr_db - reliability_penalty
        return quality_score, snr_db, speech_ratio, speech_seconds

    top_k = []
    fallback_k = []
    K = 5000
    counter = 0

    def push_candidate(heap, score, record):
        if len(heap) < K:
            heapq.heappush(heap, (score, counter, record))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, counter, record))

    print(f"🧮 Refining {len(dataset)} records...")
    for item in dataset:
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.nan_to_num(audio, copy=False)

        duration = len(audio) / sr
        if not (5.0 <= duration <= 22.0):
            continue

        audio_16k = (
            librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            if sr != TARGET_SR
            else audio
        )
        usable = (len(audio_16k) // FRAME_LEN) * FRAME_LEN
        if usable < FRAME_LEN:
            continue

        audio_16k = np.clip(audio_16k[:usable], -1.0, 1.0)
        pcm = (audio_16k * 32767).astype(np.int16).reshape(-1, FRAME_LEN)

        raw_mask = np.array(
            [vad.is_speech(frame.tobytes(), TARGET_SR) for frame in pcm],
            dtype=bool,
        )
        mask = smooth_vad_mask(raw_mask)

        if not mask.any():
            fallback_record = {
                "audio": {"array": audio, "sampling_rate": sr},
                "transcription": item["transcription"],
                "quality_score": -1_000_000.0,
                "snr_db": -999.0,
                "speech_ratio": 0.0,
                "speech_seconds": 0.0,
                "selection_tier": "fallback_no_speech",
            }
            push_candidate(fallback_k, fallback_record["quality_score"], fallback_record)
            counter += 1
            continue

        idx = np.where(mask)[0]
        frame_sec = FRAME_MS / 1000.0
        buffer_sec = 0.4

        start_sec = max(0.0, idx[0] * frame_sec - buffer_sec)
        end_sec = min(len(audio_16k) / TARGET_SR, (idx[-1] + 1) * frame_sec + buffer_sec)

        start = int(start_sec * sr)
        end = min(len(audio), int(end_sec * sr))
        trimmed_audio = audio[start:end]
        if len(trimmed_audio) == 0:
            fallback_record = {
                "audio": {"array": audio, "sampling_rate": sr},
                "transcription": item["transcription"],
                "quality_score": -900_000.0,
                "snr_db": -999.0,
                "speech_ratio": float(np.mean(mask)),
                "speech_seconds": 0.0,
                "selection_tier": "fallback_empty_trim",
            }
            push_candidate(fallback_k, fallback_record["quality_score"], fallback_record)
            counter += 1
            continue

        metrics = score_clip(audio_16k, mask)
        if metrics is None:
            fallback_record = {
                "audio": {"array": trimmed_audio, "sampling_rate": sr},
                "transcription": item["transcription"],
                "quality_score": -500_000.0 + float(np.mean(mask)),
                "snr_db": -999.0,
                "speech_ratio": float(np.mean(mask)),
                "speech_seconds": float(np.sum(mask) * (FRAME_MS / 1000.0)),
                "selection_tier": "fallback_low_confidence",
            }
            push_candidate(fallback_k, fallback_record["quality_score"], fallback_record)
            counter += 1
            continue
        quality_score, snr_db, speech_ratio, speech_seconds = metrics

        record = {
            "audio": {"array": trimmed_audio, "sampling_rate": sr},
            "transcription": item["transcription"],
            "quality_score": float(quality_score),
            "snr_db": float(snr_db),
            "speech_ratio": float(speech_ratio),
            "speech_seconds": float(speech_seconds),
            "selection_tier": "primary",
        }

        push_candidate(top_k, quality_score, record)
        counter += 1

    refined_pool = [x[2] for x in sorted(top_k, key=lambda x: x[0], reverse=True)]
    if len(refined_pool) < K:
        fallback_pool = [x[2] for x in sorted(fallback_k, key=lambda x: x[0], reverse=True)]
        needed = K - len(refined_pool)
        refined_pool.extend(fallback_pool[:needed])

    if len(refined_pool) < K:
        print(f"⚠️ Only {len(refined_pool)} clips available after fallback fill.")
    else:
        fallback_count = sum(1 for record in refined_pool if record["selection_tier"] != "primary")
        print(f"✅ Final refined pool size: {len(refined_pool)} ({fallback_count} fallback clips used)")

    top_ds = Dataset.from_list(refined_pool)

    splits = top_ds.train_test_split(test_size=1000)
    test_valid = splits["test"].train_test_split(test_size=500)

    refined_dict = DatasetDict({
        "train": splits["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]
    })

    print("💾 Saving Refined dataset to sna-refined-vol...")
    refined_dict.save_to_disk("/refined/sna_refined_v1")
    refined_vol.commit()
    
    # HF Upload removed completely!
    print("✅ Curation complete! Refined data safely stored in sna-refined-vol at /refined/sna_refined_v1")

if __name__ == "__main__":
    refine_shona_dataset.remote()