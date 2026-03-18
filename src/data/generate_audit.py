import modal
import json
from datetime import datetime

app = modal.App("sna-data-auditor")
refined_vol = modal.Volume.from_name("sna-refined-vol")

@app.function(
    image=modal.Image.debian_slim().pip_install("datasets", "numpy"),
    volumes={"/refined": refined_vol}
)
def audit_dataset():
    from datasets import load_from_disk
    import numpy as np
    
    print("📊 Loading V2 dataset for auditing...")
    ds = load_from_disk("/refined/sna_refined_v2")
    
    # Cast the PyArrow Columns to standard Python lists before concatenating
    all_snr = list(ds["train"]["snr_db"]) + list(ds["test"]["snr_db"]) + list(ds["valid"]["snr_db"])
    all_durations = list(ds["train"]["speech_seconds"]) + list(ds["test"]["speech_seconds"]) + list(ds["valid"]["speech_seconds"])
    
    stats = {
        "project": "Shona S2S Dataset",
        "stage": "Final Refined V2",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_clips": len(ds["train"]) + len(ds["test"]) + len(ds["valid"]),
        "total_speech_hours": round(sum(all_durations) / 3600, 2),
        "splits": {
            "train": len(ds["train"]),
            "test": len(ds["test"]),
            "valid": len(ds["valid"])
        },
        "acoustic_metrics": {
            "mean_snr_db": round(float(np.mean(all_snr)), 2),
            "std_snr_db": round(float(np.std(all_snr)), 2),
            "max_snr_db": round(float(np.max(all_snr)), 2),
            "min_snr_db": round(float(np.min(all_snr)), 2)
        },
        "text_normalization": "lowercase, alphanumeric + basic punctuation, smart apostrophe fusion"
    }

    print("💾 Writing curation_audit_report.json to volume...")
    with open("/refined/curation_audit_report.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    refined_vol.commit()
    print("✅ Audit complete! Here is your summary:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    audit_dataset.remote()