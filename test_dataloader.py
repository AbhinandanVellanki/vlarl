#!/usr/bin/env python3
"""
Test script to visualize RLDS dataloader contents.
Usage: python test_dataloader.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import cprint

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


def visualize_batch(batch, batch_idx=0, processor=None, save_dir="./dataloader_visualizations"):
    """Visualize a single batch from the dataloader."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"BATCH {batch_idx} CONTENTS")
    print("="*80)
    
    # Print batch keys
    print("\n📦 Batch Keys:")
    for key in batch.keys():
        if hasattr(batch[key], 'shape'):
            print(f"  - {key}: shape={batch[key].shape}, dtype={batch[key].dtype}")
        else:
            print(f"  - {key}: type={type(batch[key])}")
    
    # Extract data
    batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
    
    print(f"\n🔢 Batch Size: {batch_size}")
    
    # Iterate through samples in batch
    for i in range(min(batch_size, 3)):  # Show first 3 samples
        print(f"\n{'─'*80}")
        print(f"SAMPLE {i}")
        print(f"{'─'*80}")
        
        # 1. Show input_ids (tokenized text)
        if 'input_ids' in batch:
            input_ids = batch['input_ids'][i].numpy()
            print(f"\n📝 Input IDs (first 20 tokens): {input_ids[:20]}")
            print(f"   Total tokens: {len(input_ids)}")
            print(f"   Non-padding tokens: {(input_ids != 0).sum()}")
        
        # 2. Decode the text prompt
        if 'input_ids' in batch and processor is not None:
            try:
                # Remove padding tokens
                non_pad_tokens = input_ids[input_ids != 0]
                decoded_text = processor.tokenizer.decode(non_pad_tokens, skip_special_tokens=True)
                print(f"\n💬 Decoded Prompt:")
                print(f"   {decoded_text}")
            except Exception as e:
                print(f"\n💬 Could not decode prompt: {e}")
        
        # 3. Show action tokens
        if 'labels' in batch:
            labels = batch['labels'][i].numpy()
            print(f"\n🎯 Action Labels (first 10): {labels[:10]}")
            print(f"   Total action tokens: {len(labels)}")
            # Filter out -100 (ignore index)
            valid_actions = labels[labels != -100]
            print(f"   Valid action tokens: {len(valid_actions)}")
            if len(valid_actions) > 0:
                print(f"   Valid actions: {valid_actions}")
        
        # 4. Visualize image
        if 'pixel_values' in batch:
            pixel_values = batch['pixel_values'][i].numpy()
            print(f"\n🖼️  Image Info:")
            print(f"   Shape: {pixel_values.shape}")
            print(f"   Dtype: {pixel_values.dtype}")
            print(f"   Min/Max: {pixel_values.min():.3f} / {pixel_values.max():.3f}")
            print(f"   Mean: {pixel_values.mean():.3f}")
            
            # Convert to displayable format
            # Assuming shape is [C, H, W] and normalized
            if pixel_values.ndim == 3:
                # Transpose to [H, W, C]
                img = pixel_values.transpose(1, 2, 0)
                
                # Handle stacked channels (e.g., 6 channels for dinosiglip)
                if img.shape[-1] == 6:
                    print(f"   Note: 6 channels detected (dinosiglip), using first 3")
                    img = img[:, :, :3]
                
                # Denormalize if needed (assume [-1, 1] or [0, 1])
                if img.min() < 0:
                    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
                
                img = np.clip(img, 0, 1)
                
                # Save image
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f"Batch {batch_idx}, Sample {i}")
                plt.axis('off')
                
                img_path = os.path.join(save_dir, f"batch{batch_idx:03d}_sample{i}.png")
                plt.savefig(img_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"   ✅ Saved to: {img_path}")
        
        # 5. Show task info if available
        if 'task_id' in batch:
            task_id = batch['task_id'][i].numpy() if hasattr(batch['task_id'][i], 'numpy') else batch['task_id'][i]
            print(f"\n🎮 Task ID: {task_id}")


def main():
    """Main function to test dataloader."""
    
    # Configuration (matching your training setup)
    vla_path = "openvla/openvla-7b"  # or your fine-tuned model path
    data_root_dir = Path("./data/modified_libero_rlds")
    dataset_name = "libero_spatial_no_noops"
    
    print("="*80)
    print("RLDS DATALOADER INSPECTION SCRIPT")
    print("="*80)
    
    # Check if dataset exists
    if not data_root_dir.exists():
        cprint(f"❌ Error: Dataset directory not found: {data_root_dir}", "red")
        cprint(f"Please update the paths in this script.", "yellow")
        return
    
    print(f"\n📁 Dataset Directory: {data_root_dir}")
    print(f"📊 Dataset Name: {dataset_name}")
    
    # 1. Load processor
    print("\n" + "─"*80)
    print("STEP 1: Loading Processor")
    print("─"*80)
    
    try:
        processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        print(f"✅ Loaded processor from: {vla_path}")
        print(f"   Tokenizer vocab size: {len(processor.tokenizer)}")
        print(f"   Image processor input sizes: {processor.image_processor.input_sizes}")
    except Exception as e:
        cprint(f"❌ Error loading processor: {e}", "red")
        cprint(f"Trying to load from local path...", "yellow")
        # Try alternative path
        vla_path = "MODEL/openvla-7b-finetuned-libero-spatial"
        try:
            processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
            print(f"✅ Loaded processor from: {vla_path}")
        except:
            cprint(f"❌ Failed to load processor. Please download the model first.", "red")
            return
    
    # 2. Create batch transform
    print("\n" + "─"*80)
    print("STEP 2: Creating Batch Transform")
    print("─"*80)
    
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    print("✅ Created RLDSBatchTransform")
    
    # 3. Load dataset
    print("\n" + "─"*80)
    print("STEP 3: Loading RLDS Dataset")
    print("─"*80)
    
    try:
        vla_dataset = RLDSDataset(
            data_root_dir,
            dataset_name,
            batch_transform,
            resize_resolution=tuple(processor.image_processor.input_sizes[0][1:]),
            shuffle_buffer_size=1,  # No shuffle for inspection
            image_aug=False,  # No augmentation for inspection
        )
        print(f"✅ Loaded dataset: {dataset_name}")
        print(f"   Dataset statistics: {vla_dataset.dataset_statistics}")
    except Exception as e:
        cprint(f"❌ Error loading dataset: {e}", "red")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Iterate through dataset
    print("\n" + "─"*80)
    print("STEP 4: Iterating Through Dataset")
    print("─"*80)
    
    num_batches_to_show = 5
    print(f"\nShowing first {num_batches_to_show} batches...\n")
    
    try:
        for batch_idx, batch in enumerate(vla_dataset):
            if batch_idx >= num_batches_to_show:
                break
            
            # Visualize this batch (pass processor separately)
            visualize_batch(batch, batch_idx, processor=processor)
            
    except Exception as e:
        cprint(f"\n❌ Error during iteration: {e}", "red")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ INSPECTION COMPLETE")
    print("="*80)
    print(f"\n📂 Images saved to: ./dataloader_visualizations/")
    print("\nTo view the images:")
    print("  ls -lh dataloader_visualizations/")
    print("  # Then open the PNG files with an image viewer")


if __name__ == "__main__":
    main()

