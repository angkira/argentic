#!/usr/bin/env python3
"""Test vLLM with native Python API (not OpenAI client)."""

import httpx


def test_native_api():
    """Test vLLM /v1/completions endpoint directly."""

    print("=" * 60)
    print("Testing vLLM Native Completions API")
    print("=" * 60)

    # Test 1: Simple completion (not chat)
    print("\nTest 1: Raw completion endpoint")
    print("-" * 60)

    payload = {
        "model": "/home/angkira/models/gemma-3-4b-4bit/gemma-3n-E4B-it-quantized",
        "prompt": "<|user|>\nSay hello!<|end|>\n<|assistant|>\n",
        "max_tokens": 50,
        "temperature": 0.7,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post("http://127.0.0.1:8000/v1/completions", json=payload)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["text"]
            finish_reason = data["choices"][0]["finish_reason"]
            usage = data["usage"]

            print(f"Response: '{text}'")
            print(f"Finish reason: {finish_reason}")
            print(f"Usage: {usage}")
        else:
            print(f"Error: {response.text}")

    # Test 2: Try with even simpler prompt
    print("\nTest 2: Minimal prompt")
    print("-" * 60)

    payload2 = {
        "model": "/home/angkira/models/gemma-3-4b-4bit/gemma-3n-E4B-it-quantized",
        "prompt": "Hello",
        "max_tokens": 30,
        "temperature": 0.0,  # Greedy
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post("http://127.0.0.1:8000/v1/completions", json=payload2)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            text = data["choices"][0]["text"]
            finish_reason = data["choices"][0]["finish_reason"]
            usage = data["usage"]

            print(f"Response: '{text}'")
            print(f"Finish reason: {finish_reason}")
            print(f"Usage: {usage}")
            print(f"Response length: {len(text)} chars")
        else:
            print(f"Error: {response.text}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_native_api()
