#!/usr/bin/env python3
"""
Simple script to check RLDS dataloader prompts and images.
Usage: python test_dataloader_simple.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from termcolor import cprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


def main():
    # ============ CONFIGURATION ============
    # Update these paths to match your setup
    vla_path = "MODEL/openvla-7b-finetuned-libero-spatial"  # or "openvla/openvla-7b"
    data_root_dir = Path("./data/modified_libero_rlds")
    dataset_name = "libero_spatial_no_noops"
    save_dir = "./dataloader_samples"
    # =======================================
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("RLDS DATALOADER SIMPLE TEST")
    print("="*80)
    
    # Check paths
    if not data_root_dir.exists():
        cprint(f"❌ Dataset not found: {data_root_dir}", "red")
        return
    
    # Load processor
    print(f"\n1️⃣  Loading processor from: {vla_path}")
    try:
        processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        cprint("✅ Processor loaded", "green")
    except Exception as e:
        cprint(f"❌ Failed to load processor: {e}", "red")
        return
    
    # Create dataset
    print(f"\n2️⃣  Loading dataset: {dataset_name}")
    try:
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
        )
        
        vla_dataset = RLDSDataset(
            data_root_dir,
            dataset_name,
            batch_transform,
            resize_resolution=tuple(processor.image_processor.input_sizes[0][1:]),
            shuffle_buffer_size=1,
            image_aug=False,
        )
        cprint("✅ Dataset loaded", "green")
        print(f"   Dataset stats: {vla_dataset.dataset_statistics}")
    except Exception as e:
        cprint(f"❌ Failed to load dataset: {e}", "red")
        import traceback
        traceback.print_exc()
        return
    
    # Iterate and print samples
    print(f"\n3️⃣  Inspecting samples...")
    print("="*80)
    
    for idx, batch in enumerate(vla_dataset):
        if idx >= 5:  # Show first 5 samples
            break
        
        print(f"\n{'─'*80}")
        print(f"SAMPLE {idx}")
        print(f"{'─'*80}")
        
        # Get input_ids and decode
        input_ids = batch['input_ids'][0].numpy()
        non_pad = input_ids[input_ids != 0]
        
        try:
            prompt = processor.tokenizer.decode(non_pad, skip_special_tokens=True)
            print(f"📝 Prompt: {prompt}")
        except:
            print(f"📝 Input IDs (first 20): {input_ids[:20]}")
        
        # Get image
        pixel_values = batch['pixel_values'][0].numpy()
        print(f"🖼️  Image shape: {pixel_values.shape}")
        print(f"   Range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
        
        # Save image
        img = pixel_values.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        
        # Handle 6 channels (dinosiglip)
        if img.shape[-1] == 6:
            img = img[:, :, :3]
        
        # Normalize to [0, 1]
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        # Convert to PIL and save
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        
        img_path = os.path.join(save_dir, f"sample_{idx:03d}.png")
        pil_img.save(img_path)
        print(f"💾 Saved: {img_path}")
        
        # Get actions if available
        if 'labels' in batch:
            labels = batch['labels'][0].numpy()
            valid_actions = labels[labels != -100]
            print(f"🎯 Action tokens: {valid_actions[:10]}... ({len(valid_actions)} total)")
    
    print("\n" + "="*80)
    cprint(f"✅ Done! Images saved to: {save_dir}/", "green")
    print("="*80)


if __name__ == "__main__":
    main()


