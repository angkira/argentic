#!/usr/bin/env python3
"""
Minimal Gemma 3n E4B debug script - isolate the <pad> token issue.

Tests different input formats to find working configuration.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from pathlib import Path


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print("=" * 70)


def test_model_config():
    """Check model configuration and tokenizer details."""
    print_section("1. MODEL CONFIGURATION")

    model_id = "google/gemma-3n-E4B-it"
    print(f"Model ID: {model_id}")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    print(f"\nModel type: {config.model_type}")
    print(f"Vision config: {hasattr(config, 'vision_config')}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"\nPad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    # Check for special image tokens
    special_tokens = tokenizer.special_tokens_map
    print(f"\nSpecial tokens: {list(special_tokens.keys())}")

    # Check chat template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        print(f"\n✓ Chat template exists (length: {len(tokenizer.chat_template)})")
    else:
        print("\n⚠ No chat template found!")


def test_processor():
    """Check processor capabilities."""
    print_section("2. PROCESSOR CHECK")

    model_id = "google/gemma-3n-E4B-it"
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Processor type: {type(processor).__name__}")
    print(f"Has image_processor: {hasattr(processor, 'image_processor')}")
    print(f"Has tokenizer: {hasattr(processor, 'tokenizer')}")

    # Test image encoding
    img = Image.new("RGB", (224, 224), color="red")

    # Test 1: Direct processing
    try:
        result = processor(text="test", images=img, return_tensors="pt")
        print("\n✓ Direct processing works")
        print(f"  Keys: {list(result.keys())}")
        print(f"  input_ids shape: {result['input_ids'].shape}")
        if "pixel_values" in result:
            print(f"  pixel_values shape: {result['pixel_values'].shape}")
    except Exception as e:
        print(f"\n✗ Direct processing failed: {e}")

    # Test 2: Chat template
    try:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print("\n✓ Chat template works")
        print(f"  Generated prompt: {prompt[:200]}...")
    except Exception as e:
        print(f"\n✗ Chat template failed: {e}")


def test_text_only_generation():
    """Test text-only generation to verify model works."""
    print_section("3. TEXT-ONLY GENERATION")

    model_id = "google/gemma-3n-E4B-it"
    print("Loading model...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print(f"Model device: {model.device}")

    # Simple text prompt
    text = "What is 2+2?"
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    print(f"\nInput IDs shape: {inputs['input_ids'].shape}")
    print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

    print("\nGenerating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy for reproducibility
        )

    print(f"Output shape: {outputs.shape}")
    print(f"Output IDs: {outputs[0].tolist()}")

    # Check if all outputs are same token (pad token issue)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    unique_tokens = torch.unique(generated_ids)
    print(f"\nUnique generated tokens: {unique_tokens.tolist()}")

    decoded = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{'Result':-^70}")
    print(decoded)
    print("-" * 70)

    return processor, model


def test_image_generation(processor, model):
    """Test image + text generation."""
    print_section("4. IMAGE + TEXT GENERATION")

    # Use bird.jpg if available
    bird_path = Path("examples/bird.jpg")
    if bird_path.exists():
        img = Image.open(bird_path)
        print(f"Using: {bird_path}")
    else:
        img = Image.new("RGB", (224, 224), color="blue")
        print("Using: synthetic blue image")

    text = "Describe this image in detail."

    # Format 1: Direct processor call
    print("\n--- Format 1: Direct processor(text=..., images=...) ---")
    try:
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)

        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Input IDs: {inputs['input_ids'][0].tolist()}")
        if "pixel_values" in inputs:
            print(f"Pixel values shape: {inputs['pixel_values'].shape}")

        print("\nGenerating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        unique_tokens = torch.unique(generated_ids)
        print(f"Unique generated tokens: {unique_tokens.tolist()}")

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{'Result':-^70}")
        print(decoded)
        print("-" * 70)

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()

    # Format 2: Chat template (if supported)
    print("\n--- Format 2: Chat template ---")
    try:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ]

        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print(f"Chat template output: {prompt[:200]}...")

        # Process with image
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

        print(f"Input IDs shape: {inputs['input_ids'].shape}")

        print("\nGenerating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"\n{'Result':-^70}")
        print(decoded)
        print("-" * 70)

    except Exception as e:
        print(f"✗ Chat template failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    print("=" * 70)
    print("Gemma 3n E4B Debug Script")
    print("=" * 70)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA: Not available (will be slow!)")

    try:
        # Step 1: Check configuration
        test_model_config()

        # Step 2: Check processor
        test_processor()

        # Step 3: Test text-only
        processor, model = test_text_only_generation()

        # Step 4: Test with image
        test_image_generation(processor, model)

        print("\n" + "=" * 70)
        print("✓ Debug complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
