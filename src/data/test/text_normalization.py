import re
from pathlib import Path


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("–", " ").replace("—", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9.,?' ]", " ", text)
    text = re.sub(r"(?<=[a-z])\s*'\s*(?=[a-z])", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "unnormalized_text.txt"
    output_path = base_dir / "normalized_text.txt"

    lines = input_path.read_text(encoding="utf-8").splitlines()
    normalized_lines = [clean_text(line) for line in lines]

    output_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")

    changed = 0
    for line_no, (before, after) in enumerate(zip(lines, normalized_lines), start=1):
        if before != after:
            changed += 1
            print(f"[Line {line_no}]")
            print(f"BEFORE: {before}")
            print(f"AFTER:  {after}")
            print("-" * 60)

    print(f"Processed {len(lines)} lines.")
    print(f"Changed {changed} lines.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
