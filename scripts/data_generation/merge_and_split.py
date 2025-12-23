#!/usr/bin/env python
"""
Data Merge and Stratified Split Script
Consolidates data from multiple LLM sources and creates train/test splits.

Usage:
    python merge_and_split.py --input-dir ../data/generated --output-dir ../data/splits
"""

import json
import os
import re
import hashlib
from collections import defaultdict
from pathlib import Path
import random
import argparse

random.seed(42)  # Reproducibility


def extract_defect_type(filename: str) -> str:
    """Extract defect type from filename."""
    fname = os.path.basename(filename).lower()

    if 'edge_ring' in fname or 'edgering' in fname:
        return 'edge_ring'
    elif 'edge_loc' in fname or 'edgeloc' in fname:
        return 'edge_loc'
    elif 'center' in fname:
        return 'center'
    elif 'donut' in fname:
        return 'donut'
    elif 'scratch' in fname:
        return 'scratch'
    elif 'loc' in fname and 'edge' not in fname:
        return 'loc'
    elif 'random' in fname:
        return 'random'
    elif 'near_full' in fname or 'nearfull' in fname:
        return 'near_full'
    elif 'none' in fname or 'normal' in fname:
        return 'none'
    return 'unknown'


def load_all_samples(input_dir: Path) -> list:
    """Load all JSON samples from input directory."""
    all_samples = []

    for source_dir in input_dir.iterdir():
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        print(f"\nLoading from {source_name}...")

        for json_file in source_dir.glob("*.json"):
            # Skip merged files
            if 'training_data' in json_file.name or 'consolidated' in json_file.name:
                continue

            defect_type = extract_defect_type(json_file.name)

            with open(json_file) as f:
                data = json.load(f)

            for sample in data:
                sample['defect_type'] = defect_type
                sample['source'] = source_name
                all_samples.append(sample)

            print(f"  {json_file.name}: {len(data)} samples -> {defect_type}")

    return all_samples


def deduplicate(samples: list) -> list:
    """Remove duplicate samples based on content hash."""
    seen_hashes = set()
    unique_samples = []

    for sample in samples:
        content = sample['input'] + sample['output']
        h = hashlib.md5(content.encode()).hexdigest()

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_samples.append(sample)

    duplicates = len(samples) - len(unique_samples)
    print(f"\nDuplicates removed: {duplicates}")
    print(f"Unique samples: {len(unique_samples)}")

    return unique_samples


def stratified_split(samples: list, train_ratio: float = 0.9) -> tuple:
    """Perform stratified split by defect type."""
    by_type = defaultdict(list)
    for sample in samples:
        by_type[sample['defect_type']].append(sample)

    train_samples = []
    test_samples = []

    print("\nStratified Split:")
    for dtype in sorted(by_type.keys()):
        type_samples = by_type[dtype]
        random.shuffle(type_samples)

        split_idx = int(len(type_samples) * train_ratio)
        train_samples.extend(type_samples[:split_idx])
        test_samples.extend(type_samples[split_idx:])

        print(f"  {dtype}: {split_idx} train / {len(type_samples) - split_idx} test")

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    return train_samples, test_samples


def clean_for_training(samples: list) -> list:
    """Remove metadata fields, keep only instruction/input/output."""
    return [
        {
            'instruction': s['instruction'],
            'input': s['input'],
            'output': s['output']
        }
        for s in samples
    ]


def main():
    parser = argparse.ArgumentParser(description="Merge and split training data")
    parser.add_argument("--input-dir", type=str, default="../data/generated",
                        help="Directory with generated data")
    parser.add_argument("--output-dir", type=str, default="../data/splits",
                        help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Train split ratio (default: 0.9)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all samples
    print("=" * 60)
    print("Loading samples from all sources...")
    all_samples = load_all_samples(input_dir)
    print(f"\nTotal loaded: {len(all_samples)}")

    # Deduplicate
    print("\n" + "=" * 60)
    print("Deduplicating...")
    unique_samples = deduplicate(all_samples)

    # Show distribution
    print("\n" + "=" * 60)
    print("Samples per defect type:")
    by_type = defaultdict(int)
    for s in unique_samples:
        by_type[s['defect_type']] += 1
    for dtype in sorted(by_type.keys()):
        print(f"  {dtype}: {by_type[dtype]}")

    # Stratified split
    print("\n" + "=" * 60)
    train_samples, test_samples = stratified_split(unique_samples, args.train_ratio)
    print(f"\nFinal: {len(train_samples)} train / {len(test_samples)} test")

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving outputs...")

    # Consolidated (with metadata)
    with open(output_dir / "consolidated_all.json", "w") as f:
        json.dump(unique_samples, f, indent=2)
    print(f"  consolidated_all.json: {len(unique_samples)} samples")

    # Train/test (clean format)
    with open(output_dir / "train.json", "w") as f:
        json.dump(clean_for_training(train_samples), f, indent=2)
    print(f"  train.json: {len(train_samples)} samples")

    with open(output_dir / "test.json", "w") as f:
        json.dump(clean_for_training(test_samples), f, indent=2)
    print(f"  test.json: {len(test_samples)} samples")

    # Metadata
    metadata = {
        "total_samples": len(unique_samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_ratio": args.train_ratio,
        "defect_distribution": dict(sorted(by_type.items()))
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
