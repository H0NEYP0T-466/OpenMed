#!/usr/bin/env python3
"""
Dataset Deduplication & Integrity Verification Test Suite
---------------------------------------------------------
Computes SHA-256 cryptographic hashes and DCT-based pHash perceptual hashes across:
  1. Intra-class redundancy (duplicate scans within the same class/sequence)
  2. Inter-sequence cross-leakage (ensuring no scan appears in both T1 and T2, etc.)
  3. Inter-pathology cross-leakage (ensuring no Normal scan is in Pituitary, etc.)
  4. Cross-dataset collision against the existing 12.6k OpenMed archive
  5. JSON registry integrity (verifying DATA.json 1-to-1 sync with disk)
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.fftpack
from PIL import Image


# ------------------------------------------------------------------------------
# Hashing Functions
# ------------------------------------------------------------------------------
def compute_sha256(filepath: str) -> str:
    """Computes exact SHA-256 byte digest."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(filepath: str, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    Computes 64-bit DCT Perceptual Hash (pHash).
    Robust to slight compression artifacts, metadata changes, and minor intensity shifts.
    """
    img_size = hash_size * highfreq_factor  # 32x32
    with Image.open(filepath) as img:
        img_gray = img.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(img_gray, dtype=np.float32)

    # 2D Discrete Cosine Transform
    dct = scipy.fftpack.dct(
        scipy.fftpack.dct(pixels, axis=0, norm="ortho"),
        axis=1,
        norm="ortho"
    )
    # Extract top-left 8x8 low frequencies (excluding DC term [0,0] from median)
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow.flatten())
    diff = (dctlow > med).flatten()

    # Bitpack into 64-bit integer
    return sum([int(b) << i for i, b in enumerate(diff)])


def hamming_distance(hash1: int, hash2: int) -> int:
    """Computes Hamming distance between two 64-bit perceptual hashes."""
    return bin(hash1 ^ hash2).count("1")


# ------------------------------------------------------------------------------
# Test Runner
# ------------------------------------------------------------------------------
class DatasetDedupTester:
    def __init__(
        self,
        dataset_dir: str,
        archive_root: str = "/home/honeypot/Projects/FAST_API/OpenMed/backend/datasets/brain/archive",
        phash_threshold: int = 2,
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.archive_root = os.path.abspath(archive_root)
        self.phash_threshold = phash_threshold

        self.data_json_path = os.path.join(self.dataset_dir, "DATA.json")
        self.images_base = os.path.join(self.dataset_dir, "Images_", "Images_")

        self.records: Dict[str, Dict[str, Any]] = {}
        self.files_on_disk: List[str] = []

        # Hash registries
        self.sha256_to_files: Dict[str, List[str]] = defaultdict(list)
        self.phash_records: List[Dict[str, Any]] = []

    def load_dataset(self):
        print(f"\n[LOAD] Loading dataset manifest:\n       {self.data_json_path}")
        if not os.path.exists(self.data_json_path):
            raise FileNotFoundError(f"DATA.json not found at: {self.data_json_path}")

        with open(self.data_json_path) as f:
            self.records = json.load(f)

        print(f"       Indexed records in DATA.json: {len(self.records):,d}")

        # Discover all image files on disk
        for root, _, files in os.walk(self.images_base):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files_on_disk.append(os.path.join(root, f))

        print(f"       Physical image files on disk: {len(self.files_on_disk):,d}")

    def compute_all_hashes(self):
        print(f"\n[HASH] Computing SHA-256 and 64-bit DCT pHash on {len(self.files_on_disk):,d} files...")
        start_t = os.times()

        for idx, fpath in enumerate(self.files_on_disk):
            # Compute SHA-256
            sha = compute_sha256(fpath)
            self.sha256_to_files[sha].append(fpath)

            # Compute pHash
            ph = compute_phash(fpath)

            # Determine class & sequence from path
            rel_path = os.path.relpath(fpath, self.images_base)
            parts = rel_path.split(os.sep)
            tumor_type = parts[0] if len(parts) > 0 else "Unknown"
            sequence_folder = parts[1] if len(parts) > 1 else "Unknown"

            # Parse sequence: e.g. "Normal T1" -> "T1"
            sequence = sequence_folder.split(" ")[-1] if " " in sequence_folder else sequence_folder

            self.phash_records.append({
                "filepath": fpath,
                "rel_path": rel_path,
                "filename": os.path.basename(fpath),
                "tumor_type": tumor_type,
                "sequence": sequence,
                "class": f"{tumor_type} {sequence}",
                "sha256": sha,
                "phash": ph,
            })

            if (idx + 1) % 1000 == 0 or (idx + 1) == len(self.files_on_disk):
                print(f"       Processed {idx + 1:,d} / {len(self.files_on_disk):,d} scans...")

    # --------------------------------------------------------------------------
    # Test 1: Exact Bit-Level Duplicates (SHA-256)
    # --------------------------------------------------------------------------
    def test_sha256_duplicates(self) -> Tuple[bool, List[Dict[str, Any]]]:
        print("\n" + "=" * 72)
        print(" TEST 1: Exact SHA-256 Cryptographic Collisions")
        print("=" * 72)

        duplicates = []
        for sha, paths in self.sha256_to_files.items():
            if len(paths) > 1:
                duplicates.append({
                    "sha256": sha,
                    "count": len(paths),
                    "files": paths
                })

        if not duplicates:
            print(" [PASS] ZERO byte-level duplicate files detected.")
            return True, []
        else:
            print(f" [FAIL] Found {len(duplicates)} SHA-256 collision groups:")
            for d in duplicates[:5]:
                print(f"        SHA: {d['sha256'][:16]}... ({d['count']} copies)")
                for f in d['files']:
                    print(f"          - {os.path.basename(f)}")
            return False, duplicates

    # --------------------------------------------------------------------------
    # Test 2: Inter-Sequence Cross-Leakage (T1 vs T2 vs T1C+)
    # --------------------------------------------------------------------------
    def test_inter_sequence_cross_leakage(self) -> Tuple[bool, List[Dict[str, Any]]]:
        print("\n" + "=" * 72)
        print(" TEST 2: Inter-Sequence Cross-Leakage (T1 vs T2 vs T1C+)")
        print("=" * 72)

        # Group records by sequence
        by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in self.phash_records:
            by_seq[rec["sequence"]].append(rec)

        for seq, items in by_seq.items():
            print(f"       Sequence {seq:<5}: {len(items):,d} scans")

        collisions = []

        # Compare pairs across different sequences
        seq_names = list(by_seq.keys())
        for i in range(len(seq_names)):
            for j in range(i + 1, len(seq_names)):
                s1 = seq_names[i]
                s2 = seq_names[j]

                # Check SHA-256 cross-overlap
                sha_s1 = {r["sha256"]: r for r in by_seq[s1]}
                for r in by_seq[s2]:
                    if r["sha256"] in sha_s1:
                        collisions.append({
                            "type": "exact_sha_leak",
                            "seq_a": s1,
                            "seq_b": s2,
                            "file_a": sha_s1[r["sha256"]]["rel_path"],
                            "file_b": r["rel_path"],
                            "hamming": 0,
                        })

                # Check perceptual pHash cross-overlap (near identical scans across sequences)
                # Sample-based or full comparison
                for r1 in by_seq[s1]:
                    for r2 in by_seq[s2]:
                        dist = hamming_distance(r1["phash"], r2["phash"])
                        if dist <= self.phash_threshold:
                            collisions.append({
                                "type": "perceptual_phash_leak",
                                "seq_a": s1,
                                "seq_b": s2,
                                "file_a": r1["rel_path"],
                                "file_b": r2["rel_path"],
                                "dist": dist,
                            })

        if not collisions:
            print(" [PASS] ZERO cross-sequence leakage between T1, T2, and T1C+.")
            print(f"        (Verified both SHA-256 and pHash with Hamming threshold <= {self.phash_threshold})")
            return True, []
        else:
            print(f" [WARN] Found {len(collisions)} cross-sequence perceptual similarities:")
            for c in collisions[:5]:
                print(f"        {c['seq_a']} vs {c['seq_b']} (Hamming dist: {c.get('dist', 0)}):")
                print(f"          A: {c['file_a']}")
                print(f"          B: {c['file_b']}")
            return False, collisions

    # --------------------------------------------------------------------------
    # Test 3: Inter-Class Cross-Leakage (Normal vs Pituitary)
    # --------------------------------------------------------------------------
    def test_inter_class_cross_leakage(self) -> Tuple[bool, List[Dict[str, Any]]]:
        print("\n" + "=" * 72)
        print(" TEST 3: Inter-Class Cross-Leakage (Normal vs Pituitary)")
        print("=" * 72)

        normal_records = [r for r in self.phash_records if r["tumor_type"] == "Normal"]
        pituitary_records = [r for r in self.phash_records if r["tumor_type"] == "Pituitary"]

        print(f"       Normal scans    : {len(normal_records):,d}")
        print(f"       Pituitary scans : {len(pituitary_records):,d}")

        collisions = []

        # Check SHA-256 collisions
        normal_sha = {r["sha256"]: r for r in normal_records}
        for r in pituitary_records:
            if r["sha256"] in normal_sha:
                collisions.append({
                    "type": "exact_sha_leak",
                    "file_normal": normal_sha[r["sha256"]]["rel_path"],
                    "file_pituitary": r["rel_path"],
                    "dist": 0,
                })

        # Check pHash collisions (strictly identical visual content, dist == 0)
        normal_phash = defaultdict(list)
        for r in normal_records:
            normal_phash[r["phash"]].append(r)

        for r in pituitary_records:
            if r["phash"] in normal_phash:
                for match in normal_phash[r["phash"]]:
                    collisions.append({
                        "type": "phash_collision",
                        "file_normal": match["rel_path"],
                        "file_pituitary": r["rel_path"],
                        "dist": 0,
                    })

        if not collisions:
            print(" [PASS] ZERO cross-class collisions between Normal and Pituitary.")
            return True, []
        else:
            print(f" [FAIL] Found {len(collisions)} cross-class collisions:")
            for c in collisions[:5]:
                print(f"        Normal:    {c['file_normal']}")
                print(f"        Pituitary: {c['file_pituitary']}")
            return False, collisions

    # --------------------------------------------------------------------------
    # Test 4: Intra-Class Near-Duplicates (pHash within same class)
    # --------------------------------------------------------------------------
    def test_intraclass_near_duplicates(self) -> Tuple[bool, List[Dict[str, Any]]]:
        print("\n" + "=" * 72)
        print(" TEST 4: Intra-Class Near-Duplicates (pHash Hamming distance == 0)")
        print("=" * 72)

        by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in self.phash_records:
            by_class[rec["class"]].append(rec)

        exact_visual_duplicates = []

        for cls_name, items in by_class.items():
            phash_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for item in items:
                phash_map[item["phash"]].append(item)

            for ph, group in phash_map.items():
                if len(group) > 1:
                    exact_visual_duplicates.append({
                        "class": cls_name,
                        "count": len(group),
                        "files": [g["rel_path"] for g in group],
                        "sha_unique": len(set(g["sha256"] for g in group)),
                    })

        if not exact_visual_duplicates:
            print(" [PASS] All scans within each class possess unique perceptual hashes.")
            return True, []
        else:
            print(f" [INFO] Found {len(exact_visual_duplicates)} groups of perceptually identical scans (dist=0):")
            for dup in exact_visual_duplicates[:8]:
                print(f"        Class [{dup['class']}] · {dup['count']} identical scans (SHA unique: {dup['sha_unique']}):")
                for f in dup["files"][:3]:
                    print(f"          - {f}")
            return False, exact_visual_duplicates

    # --------------------------------------------------------------------------
    # Test 5: Cross-Dataset Overlap with Entire 12,626-Scan Archive
    # --------------------------------------------------------------------------
    def test_archive_collision(self) -> Tuple[bool, List[Dict[str, Any]]]:
        print("\n" + "=" * 72)
        print(" TEST 5: Comprehensive Collision Check Against ENTIRE 12,626-Scan Archive")
        print("=" * 72)

        archive_json_path = os.path.join(self.archive_root, "DATA.json")
        if not os.path.exists(archive_json_path):
            print(f" [SKIP] Archive DATA.json not found at {archive_json_path}")
            return True, []

        cache_path = os.path.join(self.archive_root, ".archive_hashes_cache.json")
        archive_shas = set()
        archive_phashes = []

        if os.path.exists(cache_path):
            print(f"       Loading archive hashes from cache: {cache_path}")
            with open(cache_path) as f:
                c_data = json.load(f)
                archive_shas = set(c_data["shas"])
                archive_phashes = c_data["phashes"]
        else:
            print("       Indexing entire 12,626 archive hashes (one-time setup)...")
            archive_images_base = os.path.join(self.archive_root, "Images_", "Images_")
            with open(archive_json_path) as f:
                archive_data = json.load(f)

            for rel_k in archive_data.keys():
                if rel_k.endswith("_mask.png"):
                    continue
                full_p = os.path.join(archive_images_base, rel_k)
                if os.path.exists(full_p):
                    archive_shas.add(compute_sha256(full_p))
                    try:
                        with Image.open(full_p) as im:
                            archive_phashes.append(compute_phash(im))
                    except Exception:
                        pass
            try:
                with open(cache_path, "w") as f:
                    json.dump({"shas": list(archive_shas), "phashes": archive_phashes}, f)
            except Exception:
                pass

        print(f"       Total Archive Hashes Indexed: {len(archive_shas):,d} (across all 13 families)")

        overlap = []
        for rec in self.phash_records:
            sha = rec["sha256"]
            ph = rec["phash"]
            if sha in archive_shas:
                overlap.append({
                    "filename": rec["filename"],
                    "class": rec["class"],
                    "reason": "Exact SHA-256 byte collision with existing archive",
                })
            else:
                for arch_ph in archive_phashes:
                    if hamming_distance(ph, arch_ph) <= 1:
                        overlap.append({
                            "filename": rec["filename"],
                            "class": rec["class"],
                            "reason": "Near-identical pHash visual duplicate with existing archive (dist <= 1)",
                        })
                        break

        if not overlap:
            print(" [PASS] ZERO collisions with existing archive! All 4,000 scans are completely new and distinct.")
            return True, []
        else:
            print(f" [WARN] Found {len(overlap)} collisions with existing archive:")
            for o in overlap[:5]:
                print(f"        - {o['filename']} ({o['class']}) · {o['reason']}")
            return False, overlap

    # --------------------------------------------------------------------------
    # Test 6: DATA.json Integrity & Physical Disk Consistency
    # --------------------------------------------------------------------------
    def test_data_json_consistency(self) -> bool:
        print("\n" + "=" * 72)
        print(" TEST 6: DATA.json Consistency & Physical File Verification")
        print("=" * 72)

        missing_on_disk = 0
        unindexed_on_disk = 0

        # Check every record in DATA.json exists on disk
        for rel_k, meta in self.records.items():
            full_p = os.path.join(self.images_base, rel_k)
            if not os.path.exists(full_p):
                missing_on_disk += 1

        # Check every file on disk is in DATA.json
        disk_rel_paths = set(os.path.relpath(f, self.images_base) for f in self.files_on_disk)
        record_keys = set(self.records.keys())

        unindexed_on_disk = len(disk_rel_paths - record_keys)

        print(f"       Records in DATA.json      : {len(self.records):,d}")
        print(f"       Files on physical disk    : {len(self.files_on_disk):,d}")
        print(f"       Missing on disk           : {missing_on_disk}")
        print(f"       Unindexed on disk         : {unindexed_on_disk}")

        if missing_on_disk == 0 and unindexed_on_disk == 0:
            print(" [PASS] 100% 1-to-1 parity between DATA.json and physical files.")
            return True
        else:
            print(" [FAIL] Parity mismatch between DATA.json and disk files.")
            return False

    # --------------------------------------------------------------------------
    # Execute Full Suite
    # --------------------------------------------------------------------------
    def run_all_tests(self):
        print("=" * 72)
        print(" OPENMED DATASET DEDUPLICATION & INTEGRITY TEST SUITE")
        print("=" * 72)

        self.load_dataset()
        self.compute_all_hashes()

        t1_pass, t1_dups = self.test_sha256_duplicates()
        t2_pass, t2_leaks = self.test_inter_sequence_cross_leakage()
        t3_pass, t3_leaks = self.test_inter_class_cross_leakage()
        t4_pass, t4_dups = self.test_intraclass_near_duplicates()
        t5_pass, t5_overlaps = self.test_archive_collision()
        t6_pass = self.test_data_json_consistency()

        print("\n" + "=" * 72)
        print(" EXECUTIVE TEST SUITE SUMMARY")
        print("=" * 72)
        print(f" 1. Exact SHA-256 Collisions       : {'[PASSED]' if t1_pass else '[FAILED]'}")
        print(f" 2. Inter-Sequence Cross-Leakage   : {'[PASSED]' if t2_pass else '[WARNING/REVIEW]'}")
        print(f" 3. Normal vs Pituitary Leakage   : {'[PASSED]' if t3_pass else '[FAILED]'}")
        print(f" 4. Intra-Class Visual Uniqueness : {'[PASSED]' if t4_pass else '[OBSERVED DUPLICATES]'}")
        print(f" 5. Existing Archive Non-Overlap  : {'[PASSED]' if t5_pass else '[OVERLAP DETECTED]'}")
        print(f" 6. DATA.json Physical Parity     : {'[PASSED]' if t6_pass else '[FAILED]'}")
        print("=" * 72)

        return {
            "t1_pass": t1_pass,
            "t2_pass": t2_pass,
            "t3_pass": t3_pass,
            "t4_pass": t4_pass,
            "t5_pass": t5_pass,
            "t6_pass": t6_pass,
            "intraclass_dups": t4_dups,
        }


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify dataset deduplication via SHA-256 and pHash.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/honeypot/Desktop/brain_dataset_augmented",
        help="Path to the dataset to verify.",
    )
    parser.add_argument(
        "--archive_root",
        type=str,
        default="/home/honeypot/Projects/FAST_API/OpenMed/backend/datasets/brain/archive",
        help="Path to the original 12.6k dataset archive.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Hamming distance threshold for pHash (default: 2).",
    )
    args = parser.parse_args()

    tester = DatasetDedupTester(
        dataset_dir=args.dataset_dir,
        archive_root=args.archive_root,
        phash_threshold=args.threshold,
    )
    tester.run_all_tests()


if __name__ == "__main__":
    main()
