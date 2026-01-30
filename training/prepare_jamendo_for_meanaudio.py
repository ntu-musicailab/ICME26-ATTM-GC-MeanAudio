#!/usr/bin/env python3
"""
Prepare Jamendo dataset for MeanAudio processing.

This script processes Jamendo audio files with captions from a JSON file and prepares 
them for use with the MeanAudio audio generation pipeline. It creates organized 
directory structures with audio file symlinks and TSV metadata files for training, 
validation, and testing.

Key Features:
    - Loads captions from a JSON file (format: [{"path": "subfolder/id.ext", "caption": "..."}, ...])
    - Splits data into train/val/test sets with configurable sizes
    - Creates symlinks to original audio files (preserving storage space)
    - Generates TSV files with sample IDs and captions
    - Handles missing audio files and captions gracefully
    - Uses random seed for reproducible splits

Input Requirements:
    - JSON file with captions in format: [{"path": "00/1085700.mp3", "caption": "..."}, ...]
    - Original Jamendo audio files organized in {sub_folder}/{id}_instrumental.mp3
    - Audio files should be in standard formats (mp3, wav, flac, etc.)
    - Script uses the instrumental stem from separated audio

Output Structure:
    output_dir/
    ├── train/
    │   ├── audios/
    │   │   └── {sample_id}.{ext}  (symlinks to original audio files)
    │   └── jamendo_train.tsv       (id\tcaption format)
    ├── val/
    │   ├── audios/
    │   └── jamendo_val.tsv
    ├── test/
    │   ├── audios/
    │   └── jamendo_test.tsv
    └── all/
        ├── audios/
        └── jamendo_all.tsv

TSV Format:
    Each TSV file contains two columns separated by tabs:
    - id: Unique sample identifier
    - caption: Text description of the audio

Usage:
    python prepare_jamendo_for_meanaudio.py \\
        --caption_path /path/to/jamendo_qwen.json \\
        --audio_root /path/to/Jamendo \\
        --output_dir ./data/jamendo_meanaudio_ready \\
        --val_samples 100 \\
        --test_samples 100 \\
        --seed 42

Note:
    - Samples without audio files or captions are skipped and reported
    - Symlinks are created only if they don't already exist
"""

import json
import random
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm


def load_captions_from_json(caption_path: str):
    """
    Load captions from JSON file.

    Args:
        caption_path: Path to JSON file containing captions

    Returns:
        Dictionary mapping sample_id to caption
    """
    print(f"Loading captions from {caption_path}...")
    with open(caption_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build a dictionary: sample_id -> caption
    caption_dict = {}
    for entry in data:
        file_path = entry.get("path", "")
        caption = entry.get("caption", "")

        # Extract sample ID from file path
        # Expected format: "subfolder/id.ext" (e.g., "00/1085700.mp3")
        if not file_path:
            continue

        # Extract the filename without extension as the sample ID
        # Example: "00/1085700.mp3" -> "1085700"
        filename = Path(file_path).stem
        if filename.isdigit():
            sample_id = filename
            caption_dict[sample_id] = caption
        else:
            print(
                f"Warning: Could not extract numeric sample ID from path: {file_path}"
            )

    print(f"Loaded {len(caption_dict)} captions from JSON file")
    return caption_dict


def process_split(
    sample_ids,
    caption_dict,
    audio_root: str,
    output_audio_dir: Path,
    output_tsv: Path,
    split_name: str,
):
    """
    Process a single data split.

    Args:
        sample_ids: List of sample IDs for this split
        caption_dict: Dictionary mapping sample_id to caption
        audio_root: Root path to Jamendo audio files
        output_audio_dir: Output directory for audio symlinks
        output_tsv: Output TSV file path
        split_name: Name of the split (for progress display)

    Returns:
        Tuple of (success_count, skipped_count)
    """
    # Create output directories
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Prepare TSV file
    skipped_count = 0
    success_count = 0

    with open(output_tsv, "w", encoding="utf-8") as f:
        # Write header
        f.write("id\tcaption\n")

        # Process each sample
        for sample_id in tqdm(sample_ids, desc=f"Processing {split_name} split"):
            # Get caption
            caption = caption_dict.get(sample_id, "")
            if not caption:
                print(f"Warning: No caption found for sample {sample_id}")
                skipped_count += 1
                continue

            # Get audio file path (use instrumental stem)
            sub_folder_idx = sample_id[-2:]
            audio_file = (
                Path(audio_root) / sub_folder_idx / f"{sample_id}_instrumental.mp3"
            )

            # Check if audio file exists
            if not audio_file.exists():
                print(
                    f"Warning: Audio file not found for sample {sample_id} at {audio_file}"
                )
                skipped_count += 1
                continue

            source_audio = audio_file

            # Create symlink with the sample ID as filename, preserving extension
            symlink_name = f"{sample_id}{source_audio.suffix}"
            symlink_path = output_audio_dir / symlink_name

            # Create symlink if it doesn't exist
            if not symlink_path.exists():
                try:
                    symlink_path.symlink_to(source_audio.resolve())
                except Exception as e:
                    print(f"Error creating symlink for {sample_id}: {e}")
                    skipped_count += 1
                    continue

            # Write to TSV
            # Escape tabs and newlines in caption
            caption_cleaned = (
                caption.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            )
            f.write(f"{sample_id}\t{caption_cleaned}\n")
            success_count += 1

    return success_count, skipped_count


def prepare_jamendo_dataset(
    caption_path: str,
    audio_root: str,
    output_dir: str,
    val_samples: int,
    test_samples: int,
    seed: int = 42,
):
    """
    Prepare Jamendo dataset for MeanAudio extraction pipeline.
    Creates train/val/test/all splits.

    Args:
        caption_path: Path to JSON file containing captions
        audio_root: Root path to Jamendo audio files
        output_dir: Base output directory
        val_samples: Number of samples for validation split
        test_samples: Number of samples for test split
        seed: Random seed for reproducibility
    """
    # Load captions from JSON
    caption_dict = load_captions_from_json(caption_path)

    # Filter sample IDs to only include those with existing audio files
    print(f"\nChecking audio file existence...")
    audio_root_path = Path(audio_root)
    valid_sample_ids = []
    missing_count = 0

    for sample_id in tqdm(caption_dict.keys(), desc="Validating audio files"):
        sub_folder_idx = sample_id[-2:]
        audio_file = audio_root_path / sub_folder_idx / f"{sample_id}_instrumental.mp3"
        if audio_file.exists():
            valid_sample_ids.append(sample_id)
        else:
            missing_count += 1

    print(f"Total samples in caption file: {len(caption_dict)}")
    print(f"Samples with existing audio files: {len(valid_sample_ids)}")
    print(f"Samples with missing audio files: {missing_count}")

    if len(valid_sample_ids) == 0:
        print("Error: No valid audio files found!")
        return

    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(valid_sample_ids)

    # Split sample IDs
    test_sample_ids = valid_sample_ids[:test_samples]
    val_sample_ids = valid_sample_ids[test_samples : test_samples + val_samples]
    train_sample_ids = valid_sample_ids[test_samples + val_samples :]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_sample_ids)} samples")
    print(f"  Val:   {len(val_sample_ids)} samples")
    print(f"  Test:  {len(test_sample_ids)} samples")
    print(f"  All:   {len(valid_sample_ids)} samples")

    # Process each split
    output_base = Path(output_dir)
    splits = {
        "train": train_sample_ids,
        "val": val_sample_ids,
        "test": test_sample_ids,
        "all": valid_sample_ids,
    }

    total_success = 0
    total_skipped = 0

    for split_name, sample_ids in splits.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split...")
        print(f"{'=' * 60}")

        split_audio_dir = output_base / split_name / "audios"
        split_tsv = output_base / split_name / f"jamendo_{split_name}.tsv"

        success, skipped = process_split(
            sample_ids, caption_dict, audio_root, split_audio_dir, split_tsv, split_name
        )

        print(
            f"{split_name.capitalize()} split: {success} successful, {skipped} skipped"
        )
        print(f"  Audio symlinks: {split_audio_dir}")
        print(f"  TSV file: {split_tsv}")

        total_success += success
        total_skipped += skipped

    print(f"\n{'=' * 60}")
    print(f"Dataset preparation completed!")
    print(f"{'=' * 60}")
    print(f"Total successfully processed: {total_success} entries")
    print(f"Total skipped: {total_skipped} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Jamendo dataset for MeanAudio processing with train/val/test splits"
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
        help="Root path to Jamendo audio files",
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="./data/caption/jamendo_qwen.json",
        help='Path to JSON file containing captions (format: [{"path": "00/1085700.mp3", "caption": "..."}, ...])',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/jamendo_meanaudio_ready",
        help="Base output directory (will create train/val/test/all subdirs)",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=100,
        help="Number of samples for validation split",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=100,
        help="Number of samples for test split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    prepare_jamendo_dataset(
        caption_path=args.caption_path,
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
