#!/usr/bin/env python3
"""Direct test of vLLM vision API to debug empty responses."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image


async def test_vision():
    """Test vLLM vision API directly."""

    # Load image - use very small size for testing
    image_path = Path("examples/bird.jpg")
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        return

    img = Image.open(image_path)
    img.thumbnail((224, 224), Image.Resampling.LANCZOS)  # Smaller for testing

    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    img_url = f"data:image/png;base64,{img_b64}"

    print("=" * 60)
    print("Testing vLLM Vision API")
    print("=" * 60)
    print(f"Image size: {img.size}")
    print(f"Image data length: {len(img_b64)} bytes")
    print()

    # Create client
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy",
        timeout=120.0,
    )

    # Get model name
    models = await client.models.list()
    model_name = models.data[0].id
    print(f"Model: {model_name}")
    print()

    # Test 1: Simple text-only request with logprobs to see tokens
    print("Test 1: Text-only request (with logprobs)")
    print("-" * 60)
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say hello in 5 words or less!"}
            ],
            max_tokens=20,
            temperature=0.7,
            logprobs=True,
            top_logprobs=1,
        )
        content = response.choices[0].message.content
        print(f"Response: '{content}'")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Usage: {response.usage}")

        # Check what tokens were generated
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            print("\nGenerated tokens:")
            for i, token_data in enumerate(response.choices[0].logprobs.content[:10]):  # First 10
                print(f"  {i}: token='{token_data.token}' (bytes={token_data.bytes})")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Test 2: Vision request with SHORT prompt
    print("Test 2: Vision request (short prompt)")
    print("-" * 60)
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            max_tokens=512,  # Much higher
            temperature=0.7,
        )
        content = response.choices[0].message.content
        print(f"Response: '{content}'")
        print(f"Finish reason: {response.choices[0].finish_reason}")
        print(f"Response length: {len(content) if content else 0} chars")
        print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    print()

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_vision())
