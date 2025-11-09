#!/usr/bin/env python3
"""
Script to explain the "Chinese characters" in action labels.
These are actually tokenized robot actions!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor

# Load a tokenizer
print("="*70)
print("Understanding Action Tokenization")
print("="*70)

vla_path = "MODEL/openvla-7b-finetuned-libero-spatial"
print(f"\nLoading tokenizer from: {vla_path}")

try:
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    print("✓ Loaded\n")
except:
    print("✗ Failed to load. Update the path in this script.\n")
    sys.exit(1)

# Example: What happens when we tokenize and decode actions
print("─"*70)
print("Example 1: Normal Text")
print("─"*70)

normal_text = "In: What action should the robot take to pick up the mug?\nOut:"
tokens = processor.tokenizer.encode(normal_text)
decoded = processor.tokenizer.decode(tokens)

print(f"Input:   {normal_text}")
print(f"Tokens:  {tokens[:10]}... ({len(tokens)} total)")
print(f"Decoded: {decoded}")

print("\n" + "─"*70)
print("Example 2: Action Tokens (the 'Chinese' characters)")
print("─"*70)

# Simulate action tokens (these would come from quantized robot actions)
action_token_ids = [32100, 32101, 32102, 32103, 32104, 32105, 32106, 32107]

print(f"\nAction token IDs: {action_token_ids}")
print("These represent quantized robot actions like:")
print("  [x, y, z, roll, pitch, yaw, gripper, ...]")

# Decode them as if they were text
decoded_actions = processor.tokenizer.decode(action_token_ids, skip_special_tokens=False)
print(f"\nWhen decoded as text: {repr(decoded_actions)}")
print("                       ^ These look like 'Chinese' but are just Unicode!")

print("\n" + "─"*70)
print("Explanation")
print("─"*70)

print("""
Why do actions look like random characters?

1. Robot actions are continuous values: [0.123, -0.456, 0.789, ...]

2. OpenVLA quantizes these into discrete tokens: [32100, 32101, ...]
   - These token IDs are added to the vocabulary
   - They're treated like any other token in the LLM

3. When you DECODE these token IDs as text:
   - The tokenizer maps them to Unicode characters
   - Since they're special tokens, they map to unused Unicode ranges
   - This creates what looks like Chinese/Arabic/random scripts

4. During training:
   - The model learns to predict these action tokens
   - They're treated as regular tokens in the sequence
   - Loss is computed on these tokens like any other text

5. During inference:
   - Model generates action tokens
   - These are DE-TOKENIZED back to continuous values
   - Then sent to the robot

So: Actions → Tokens → "Chinese" (when decoded) → Actions (when used)
""")

print("─"*70)
print("What you see in the dataset:")
print("─"*70)

example_prompt = """
In: What action should the robot take to pick up the black bowl?
Out: 貴飛庄ĦĦਿ忠</s>
     ^^^^^^^^^
     These are the action tokens decoded as Unicode!
"""
print(example_prompt)

print("\n" + "="*70)
print("Summary")
print("="*70)
print("""
✓ The 'Chinese characters' are NORMAL and EXPECTED
✓ They are robot actions tokenized into the vocabulary
✓ The model learns to predict these special tokens
✓ During inference, they're converted back to robot actions

Don't worry about them - they're just how actions are represented!
""")

