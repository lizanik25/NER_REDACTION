import argparse
import os
import urllib.request
from pathlib import Path


NAVEC_URL = (
    "https://github.com/natasha/navec/releases/download/v0.10/"
    "navec_hudlit_v1_12B_500K_250d_100q.tar"
)
NAVEC_FILENAME = "navec_hudlit_v1_12B_500K_250d_100q.tar"

SLOVNET_RELEASE_URL = (
    "https://github.com/lizanik25/NER_REDACTION/releases/download/v1.0/"
    "slovnet_ner.bin"
)
SLOVNET_FILENAME = "slovnet_ner.bin"


def download_file(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading {desc or url}")
    print(f"  -> {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print(f"\r  Done ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--model_dir",
        default="models/final_model",
        help="Directory to save model files",
    )
    parser.add_argument(
        "--skip_navec",
        action="store_true",
        help="Skip downloading Navec embeddings",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_navec:
        download_file(
            NAVEC_URL,
            model_dir / NAVEC_FILENAME,
            desc="Navec embeddings",
        )
    else:
        print("  Skipping Navec")

    download_file(
        SLOVNET_RELEASE_URL,
        model_dir / SLOVNET_FILENAME,
        desc="Slovnet NER weights (fine-tuned)",
    )

    print(f"Model directory: {model_dir.resolve()}")
    print("\nFiles:")
    for f in sorted(model_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
