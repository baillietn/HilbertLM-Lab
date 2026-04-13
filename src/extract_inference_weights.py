import torch
import argparse
import os
from pathlib import Path


def extract_inference_weights(checkpoint_path: str, output_path: str = None, quiet: bool = False):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not quiet:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if output_path is None:
        base_path = Path(checkpoint_path)
        output_path = base_path.parent / f"{base_path.stem}_inference.pt"
    
    if not quiet:
        print(f"Checkpoint keys: {checkpoint.keys()}")
    inference_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict']
    }
    original_size = sum(
        p.numel() for p in checkpoint['model_state_dict'].values() 
        if isinstance(p, torch.Tensor)
    )
    
    stats = {
        'original_checkpoint': checkpoint_path,
        'inference_checkpoint': str(output_path),
        'model_params': original_size,
        'keys_kept': list(inference_checkpoint.keys())
    }
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer_params = sum(
            p.numel() for p in checkpoint['optimizer_state_dict'].values() 
            if isinstance(p, torch.Tensor)
        )
        stats['optimizer_params_removed'] = optimizer_params
        if not quiet:
            print(f"Removed optimizer state with {optimizer_params} parameters")
    
    if 'history' in checkpoint:
        stats['history_removed'] = True
        if not quiet:
            print("Removed training history")
    
    if 'samples_seen' in checkpoint:
        stats['samples_seen_removed'] = checkpoint['samples_seen']
        if not quiet:
            print(f"Removed training metadata (samples_seen: {checkpoint['samples_seen']})")
    torch.save(inference_checkpoint, output_path)
    original_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    inference_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    size_reduction = ((original_size_mb - inference_size_mb) / original_size_mb) * 100
    
    if not quiet:
        print(f"\n✓ Successfully extracted inference weights!")
        print(f"  Original checkpoint size: {original_size_mb:.2f} MB")
        print(f"  Inference checkpoint size: {inference_size_mb:.2f} MB")
        print(f"  Size reduction: {size_reduction:.1f}%")
        print(f"  Saved to: {output_path}")
    
    stats['original_size_mb'] = original_size_mb
    stats['inference_size_mb'] = inference_size_mb
    stats['size_reduction_percent'] = size_reduction
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract inference-only weights from a checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint file (e.g., checkpoints/hilbert_chat_model.pt)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for inference checkpoint. If not specified, adds '_inference' suffix"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    stats = extract_inference_weights(
        args.checkpoint,
        args.output,
        args.quiet
    )
