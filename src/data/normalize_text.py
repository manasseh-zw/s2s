import modal
import re

app = modal.App("sna-data-normalizer")
refined_vol = modal.Volume.from_name("sna-refined-vol")

@app.function(
    image=modal.Image.debian_slim().pip_install("datasets"),
    volumes={"/refined": refined_vol}
)
def normalize_dataset():
    from datasets import load_from_disk

    print("📝 Loading V1 dataset from volume...")
    ds = load_from_disk("/refined/sna_refined_v1")

    def clean_text(example):
        text = str(example["transcription"]).lower()

        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = text.replace("–", " ").replace("—", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9.,?' ]", " ", text)
        text = re.sub(r"(?<=[a-z])\s*'\s*(?=[a-z])", "'", text)
        text = re.sub(r"\s+", " ", text).strip()

        example["transcription"] = text
        return example

    print("✨ Applying 'Light Touch' Shona text normalization...")
    ds_v2 = ds.map(clean_text)

    print("💾 Saving as V2 to volume...")
    ds_v2.save_to_disk("/refined/sna_refined_v2")
    refined_vol.commit()

    print("✅ Normalization complete! V2 safely stored at /refined/sna_refined_v2.")

    print("-" * 50)
    print(f"🔍 SAMPLE BEFORE: {ds['train'][0]['transcription']}")
    print(f"🔍 SAMPLE AFTER:  {ds_v2['train'][0]['transcription']}")
    print("-" * 50)

if __name__ == "__main__":
    normalize_dataset.remote()