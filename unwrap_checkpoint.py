#!/usr/bin/env python3
"""
Script to unwrap model state dict by removing the 'ema_model.' prefix from all keys.
"""

import argparse
import torch
from pathlib import Path


def unwrap_state_dict(state_dict, prefix="ema_model."):
    """
    Remove the specified prefix from all keys in the state dict.
    Only keeps keys that have the prefix, removing all others.

    Args:
        state_dict: Dictionary containing model weights
        prefix: The prefix to remove (default: "ema_model.")

    Returns:
        Dictionary with unwrapped keys (only keys that had the prefix)
    """
    unwrapped = {}
    prefix_len = len(prefix)

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            unwrapped[new_key] = value
        # Skip keys that don't have the prefix

    return unwrapped


def main():
    parser = argparse.ArgumentParser(
        description="Unwrap model checkpoint by removing 'ema_model.' prefix from state dict keys"
    )
    parser.add_argument(
        "input_path", type=str, help="Path to input .pt checkpoint file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output .pt file (default: input_path with '_unwrapped' suffix)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="ema_model.",
        help="Prefix to remove from keys (default: 'ema_model.')",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without saving"
    )

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.input_path}")
    checkpoint = torch.load(args.input_path, map_location="cpu")

    # Check if checkpoint is a state dict or contains a state dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("Found 'state_dict' key in checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint as state dict directly")

    # Show sample keys before unwrapping
    sample_keys = list(state_dict.keys())[:5]
    print(f"\nSample keys before unwrapping:")
    for key in sample_keys:
        print(f"  {key}")

    # Unwrap the state dict
    unwrapped_state_dict = unwrap_state_dict(state_dict, prefix=args.prefix)

    # Show sample keys after unwrapping
    sample_keys = list(unwrapped_state_dict.keys())[:5]
    print(f"\nSample keys after unwrapping:")
    for key in sample_keys:
        print(f"  {key}")

    print(f"\nTotal keys: {len(unwrapped_state_dict)}")

    if args.dry_run:
        print("\nDry run - not saving output")
        return

    # Determine output path
    if args.output is None:
        input_path = Path(args.input_path)
        output_path = (
            input_path.parent / f"{input_path.stem}_unwrapped{input_path.suffix}"
        )
    else:
        output_path = Path(args.output)

    # Save unwrapped checkpoint
    print(f"\nSaving unwrapped checkpoint to: {output_path}")

    # If original checkpoint had other keys, preserve them
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint["state_dict"] = unwrapped_state_dict
        torch.save(checkpoint, output_path)
    else:
        torch.save(unwrapped_state_dict, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
