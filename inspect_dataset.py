#!/usr/bin/env python3
"""
Minimal script to inspect RLDS dataset contents.
Prints prompts and saves images.

Usage: python inspect_dataset.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# ============ CONFIGURATION ============
VLA_PATH = "MODEL/openvla-7b-finetuned-libero-spatial"
DATA_ROOT = "./data/modified_libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"
NUM_SAMPLES = 3
SAVE_DIR = "./dataset_inspection"
# =======================================


def print_colored(text, color_code):
    """Simple colored printing."""
    print(f"\033[{color_code}m{text}\033[0m")


def save_image(pixel_values, save_path):
    """Save image tensor as PNG."""
    # pixel_values shape: [C, H, W]
    img = pixel_values.transpose(1, 2, 0)  # -> [H, W, C]
    
    # Handle 6 channels (dinosiglip stacks)
    if img.shape[-1] == 6:
        img = img[:, :, :3]
    
    # Denormalize: assume [-1, 1] or [0, 1]
    if img.min() < 0:
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    pil_img.save(save_path)
    return img_uint8.shape


def main():
    print("="*70)
    print("RLDS Dataset Inspector")
    print("="*70)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Import after adding to path
    from transformers import AutoProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    
    # 1. Load processor
    print(f"\n[1/3] Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)
        print_colored(f"✓ Loaded from {VLA_PATH}", "92")
    except Exception as e:
        print_colored(f"✗ Error: {e}", "91")
        return
    
    # 2. Setup dataset
    print(f"\n[2/3] Setting up dataset...")
    try:
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        
        dataset = RLDSDataset(
            Path(DATA_ROOT),
            DATASET_NAME,
            batch_transform,
            resize_resolution=tuple(processor.image_processor.input_sizes[0][1:]),
            shuffle_buffer_size=1,
            image_aug=False,
        )
        print_colored(f"✓ Dataset: {DATASET_NAME}", "92")
        
        if hasattr(dataset, 'dataset_statistics'):
            print(f"  Stats: {dataset.dataset_statistics}")
    except Exception as e:
        print_colored(f"✗ Error: {e}", "91")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Inspect samples
    print(f"\n[3/3] Inspecting {NUM_SAMPLES} samples...")
    print("="*70)
    
    for idx, batch in enumerate(dataset):
        if idx >= NUM_SAMPLES:
            break
        
        print(f"\n{'─'*70}")
        print(f"Sample {idx}")
        print(f"{'─'*70}")
        
        # Decode prompt
        input_ids = batch['input_ids'][0].numpy()
        non_padding = input_ids[input_ids != 0]
        
        try:
            prompt = processor.tokenizer.decode(non_padding, skip_special_tokens=True)
            print(f"\nPrompt:")
            print(f"  \"{prompt}\"")
        except Exception as e:
            print(f"  [Could not decode: {e}]")
            print(f"  Token IDs (first 20): {input_ids[:20]}")
        
        # Image info
        pixel_values = batch['pixel_values'][0].numpy()
        print(f"\nImage:")
        print(f"  Shape: {pixel_values.shape}")
        print(f"  Dtype: {pixel_values.dtype}")
        print(f"  Range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
        
        # Save image
        img_path = os.path.join(SAVE_DIR, f"sample_{idx}.png")
        img_shape = save_image(pixel_values, img_path)
        print(f"  Saved: {img_path} (shape: {img_shape})")
        
        # Action labels
        if 'labels' in batch:
            labels = batch['labels'][0].numpy()
            valid_labels = labels[labels != -100]
            print(f"\nActions:")
            print(f"  Total tokens: {len(labels)}")
            print(f"  Valid tokens: {len(valid_labels)}")
            if len(valid_labels) > 0:
                print(f"  First 10: {valid_labels[:10]}")
        
        # Additional info
        print(f"\nBatch keys: {list(batch.keys())}")
    
    print("\n" + "="*70)
    print_colored(f"✓ Complete! Saved to: {SAVE_DIR}/", "92")
    print("="*70)
    
    # Print summary
    print("\nTo view images:")
    print(f"  ls -lh {SAVE_DIR}/")
    print(f"  # Or open *.png files with image viewer")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\033[91mUnexpected error: {e}\033[0m")
        import traceback
        traceback.print_exc()


