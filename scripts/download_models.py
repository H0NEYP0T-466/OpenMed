"""
OpenMed - 3D Organ Model Downloader
Fetches verified anatomical GLB models from the NIH / Human Reference Atlas (HRA) library.
"""

import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://raw.githubusercontent.com/hubmapconsortium/ccf-3d-reference-object-library/main"

MODELS = {
    "brain.glb": f"{BASE_URL}/VH_Male/v1.2/Allen_M_Brain.glb",
    "lungs.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Lung.glb",
    "heart.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Heart.glb",
    "kidney.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Kidney_L.glb",
    "eye.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Eye_L.glb",
    "blood.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Blood_Vasculature.glb",
    "liver.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Liver.glb",
    "bone.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Vertebrae.glb",
    "breast.glb": f"{BASE_URL}/VH_Female/v1.4/VHF_extraction_landmarks/3d-vh-f-mammary-gland-l-landmark.glb",
    "skin.glb": f"{BASE_URL}/VH_Male/v1.2/VH_M_Skin.glb",
    "human_body.glb": f"{BASE_URL}/VH_Male/v1.1/VH_M_United.glb",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "models", "organs")


def download_model(filename: str, url: str) -> tuple[str, bool, str]:
    dest_path = os.path.join(OUTPUT_DIR, filename)
    
    # Check if valid file already exists
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        with open(dest_path, "rb") as f:
            if f.read(4) == b"glTF":
                return (filename, True, f"Already cached ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")

    temp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenMed-FYP-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(temp_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            chunk_size = 1024 * 64
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
        
        # Verify glTF header
        with open(temp_path, "rb") as f:
            magic = f.read(4)
            if magic != b"glTF":
                os.remove(temp_path)
                return (filename, False, f"Invalid glTF header: {magic}")
        
        os.replace(temp_path, dest_path)
        final_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        return (filename, True, f"Downloaded ({final_size_mb:.2f} MB)")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return (filename, False, str(e))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Target directory: {OUTPUT_DIR}")
    print(f"Starting parallel download for {len(MODELS)} anatomical models...")

    failed = False
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_model, name, url): name for name, url in MODELS.items()}
        for future in as_completed(futures):
            name, success, msg = future.result()
            symbol = "✓" if success else "✗"
            print(f"  [{symbol}] {name:16}: {msg}")
            if not success:
                failed = True

    if failed:
        print("\nSome models failed to download.")
        sys.exit(1)
    else:
        print("\nAll 3D models downloaded and verified successfully!")


if __name__ == "__main__":
    main()
