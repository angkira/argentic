#!/usr/bin/env python3
"""Test vLLM offline API (like HuggingFace example)."""

from vllm import LLM, SamplingParams
from pathlib import Path

def test_offline():
    """Test vLLM offline (non-server) mode."""

    print("=" * 60)
    print("Testing vLLM Offline API")
    print("=" * 60)

    model_path = str(Path.home() / "models/gemma-3-4b-4bit/gemma-3n-E4B-it-quantized")

    print(f"\nLoading model: {model_path}")
    print("This may take a minute...")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,  # Reduced from 4096
        gpu_memory_utilization=0.7,  # Reduced from 0.9
        max_num_seqs=1,  # Reduce batch size
    )

    print("✓ Model loaded\n")

    # Test 1: Simple text generation
    print("Test 1: Simple text generation")
    print("-" * 60)

    prompts = ["<|user|>\nSay hello!<|end|>\n<|assistant|>\n"]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: '{generated_text}'")
        print(f"Length: {len(generated_text)} chars")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_offline()
