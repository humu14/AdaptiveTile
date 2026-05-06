"""Downloads Kodak dataset (24 images). DIV2K and BSDS500 require manual download."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from tqdm import tqdm
from src.config import KODAK_DIR, DIV2K_DIR, BSDS500_DIR

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
KODAK_COUNT = 24


def download_kodak():
    KODAK_DIR.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(1, KODAK_COUNT + 1), desc="Downloading Kodak"):
        name = f"kodim{i:02d}.png"
        dest = KODAK_DIR / name
        if dest.exists():
            continue
        url = f"{KODAK_BASE}/{name}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
    print(f"Kodak: {KODAK_COUNT} images in {KODAK_DIR}")


def print_manual_instructions():
    print("\n--- DIV2K (manual) ---")
    print("1. Visit: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
    print("2. Download DIV2K_valid_HR.zip")
    print(f"3. Extract to: {DIV2K_DIR}/valid_HR/")

    print("\n--- BSDS500 (manual) ---")
    print("1. Visit: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/")
    print("2. Download BSR_bsds500.tgz")
    print(f"3. Extract images/test/ to: {BSDS500_DIR}/images/test/")


if __name__ == "__main__":
    download_kodak()
    print_manual_instructions()
