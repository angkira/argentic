#!/usr/bin/env python3
"""Investigate the model's tokenizer and generation config."""

from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
import json

model_path = str(Path.home() / "models/gemma-3-4b-4bit/gemma-3n-E4B-it-quantized")

print("=" * 60)
print("Investigating Model Configuration")
print("=" * 60)

# Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"   Tokenizer class: {tokenizer.__class__.__name__}")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Model max length: {tokenizer.model_max_length}")
print(f"   BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"   UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

# Test tokenization
print("\n2. Testing tokenization...")
test_texts = [
    "Hello world",
    "<|user|>\nHello!<|end|>\n<|assistant|>\n",
    "The quick brown fox"
]

for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"   Text: '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: '{decoded}'")
    print()

# Check generation config
print("3. Checking generation config...")
try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, 'generation_config'):
        gen_config = config.generation_config
        print("   Generation config found:")
        print(f"   {json.dumps(gen_config.to_dict(), indent=4)}")
    else:
        print("   No generation config in model config")
except Exception as e:
    print(f"   Error loading config: {e}")

# Check for generation_config.json
print("\n4. Checking for generation_config.json...")
gen_config_path = Path(model_path) / "generation_config.json"
if gen_config_path.exists():
    with open(gen_config_path) as f:
        gen_config_data = json.load(f)
    print("   Found generation_config.json:")
    print(f"   {json.dumps(gen_config_data, indent=4)}")
else:
    print("   No generation_config.json found")

# Check chat template
print("\n5. Checking chat template...")
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    print("   Chat template exists:")
    print(f"   {tokenizer.chat_template[:500]}")  # First 500 chars
else:
    print("   No chat template")

# Test chat template
print("\n6. Testing chat template...")
try:
    messages = [
        {"role": "user", "content": "Say hello!"}
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"   Input messages: {messages}")
    print(f"   Formatted text: '{chat_text}'")

    # Tokenize and decode
    tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    decoded = tokenizer.decode(tokens)
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: '{decoded}'")
except Exception as e:
    print(f"   Error applying chat template: {e}")

print("\n" + "=" * 60)
