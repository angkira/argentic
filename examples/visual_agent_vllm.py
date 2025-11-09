c  #!/usr/bin/env python3
"""Visual Agent example using vLLM provider with Gemma 3n."""

import argparse
import asyncio
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
from PIL import Image

from argentic import Messager
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.llm_factory import LLMFactory
from argentic.core.tools import ToolManager


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single query."""

    frame_count: int
    query: str
    response: str
    response_length: int  # characters
    total_time: float  # seconds
    connect_time: float  # seconds
    query_time: float  # seconds
    # Token tracking - ACTUAL from API
    actual_prompt_tokens: Optional[int] = None  # From API
    actual_completion_tokens: Optional[int] = None  # From API
    actual_total_tokens: Optional[int] = None  # From API
    # Token tracking - ESTIMATED
    estimated_prompt_tokens: Optional[int] = None
    estimated_image_tokens: Optional[int] = None
    estimated_output_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None  # Output tokens/sec
    # Memory tracking
    memory_used_mb: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None


@dataclass
class BenchmarkStats:
    """Statistics for multiple benchmark iterations."""

    frame_count: int
    iterations: int
    # Timing stats
    mean_query_time: float
    std_query_time: float
    min_query_time: float
    max_query_time: float
    # Token stats (actual)
    mean_prompt_tokens: Optional[float] = None
    mean_completion_tokens: Optional[float] = None
    mean_total_tokens: Optional[float] = None
    # Throughput stats
    mean_tokens_per_second: Optional[float] = None
    std_tokens_per_second: Optional[float] = None
    # All individual metrics for detailed analysis
    all_metrics: list[PerformanceMetrics] = None


# Mock WebRTC driver for testing
class MockWebRTCDriver:
    """Mock driver that returns static images (can be multiple different images)."""

    def __init__(self, image_paths: list[Path], video_buffer_size: int = 1):
        """Initialize driver with list of image paths.

        Args:
            image_paths: List of paths to images. If buffer needs more frames than images,
                        images will be cycled.
            video_buffer_size: Number of frames to create
        """
        self.image_paths = (
            image_paths if isinstance(image_paths, list) else [image_paths]
        )
        self.video_buffer_size = video_buffer_size
        self._frames = None
        self._connected = False
        self._capturing = False

    async def connect(self, offer_sdp: str = None):
        """Load images and create frames."""
        import numpy as np

        self._connected = True

        # Load all unique images up to video_buffer_size
        frames = []
        for i in range(self.video_buffer_size):
            # Cycle through available images
            img_path = self.image_paths[i % len(self.image_paths)]
            img = Image.open(img_path)
            # Resize to max 640px for faster processing
            img.thumbnail((640, 640), Image.Resampling.LANCZOS)
            img_array = np.array(img.convert("RGB"))
            frames.append(img_array)

        self._frames = frames
        return None

    async def start_capture(self):
        self._capturing = True

    async def stop_capture(self):
        self._capturing = False

    async def disconnect(self):
        self._connected = False
        self._frames = None

    async def get_frame_buffer(self):
        if self._frames:
            return self._frames
        return []

    async def get_audio_buffer(self):
        return None

    def clear_buffers(self):
        pass


def calculate_benchmark_stats(metrics_list: list[PerformanceMetrics]) -> BenchmarkStats:
    """Calculate statistics from multiple benchmark runs.

    Args:
        metrics_list: List of PerformanceMetrics from multiple iterations

    Returns:
        BenchmarkStats with mean, std, min, max values
    """
    import numpy as np

    if not metrics_list:
        raise ValueError("Empty metrics list")

    frame_count = metrics_list[0].frame_count
    iterations = len(metrics_list)

    # Extract timing data
    query_times = [m.query_time for m in metrics_list]
    mean_query_time = np.mean(query_times)
    std_query_time = np.std(query_times)
    min_query_time = np.min(query_times)
    max_query_time = np.max(query_times)

    # Extract token data (if available)
    prompt_tokens = [
        m.actual_prompt_tokens
        for m in metrics_list
        if m.actual_prompt_tokens is not None
    ]
    completion_tokens = [
        m.actual_completion_tokens
        for m in metrics_list
        if m.actual_completion_tokens is not None
    ]
    total_tokens = [
        m.actual_total_tokens for m in metrics_list if m.actual_total_tokens is not None
    ]

    mean_prompt_tokens = np.mean(prompt_tokens) if prompt_tokens else None
    mean_completion_tokens = np.mean(completion_tokens) if completion_tokens else None
    mean_total_tokens = np.mean(total_tokens) if total_tokens else None

    # Extract throughput data
    tps_values = [
        m.tokens_per_second for m in metrics_list if m.tokens_per_second is not None
    ]
    mean_tokens_per_second = np.mean(tps_values) if tps_values else None
    std_tokens_per_second = np.std(tps_values) if tps_values else None

    return BenchmarkStats(
        frame_count=frame_count,
        iterations=iterations,
        mean_query_time=mean_query_time,
        std_query_time=std_query_time,
        min_query_time=min_query_time,
        max_query_time=max_query_time,
        mean_prompt_tokens=mean_prompt_tokens,
        mean_completion_tokens=mean_completion_tokens,
        mean_total_tokens=mean_total_tokens,
        mean_tokens_per_second=mean_tokens_per_second,
        std_tokens_per_second=std_tokens_per_second,
        all_metrics=metrics_list,
    )


def estimate_text_tokens(text: str) -> int:
    """Estimate number of tokens in text.

    Uses a simple heuristic: ~4 characters per token for English text.
    For more accurate counting, you could use tiktoken or similar.
    """
    return max(1, len(text) // 4)


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate tokens used by an image.

    For vision models like Gemma 3n, images are typically processed in patches.
    Common approach: 336x336 patches, each patch ~170 tokens.
    This is an approximation - actual token count depends on model architecture.

    Gemma 3n (PaliGemma) uses ~256 image tokens per image regardless of size
    after resizing to 224x224 or similar.
    """
    # Conservative estimate: ~256 tokens per image for PaliGemma-style models
    # For higher resolution models, this could be much higher
    return 256


def get_gpu_memory() -> Optional[float]:
    """Get GPU memory usage in MB using nvidia-smi if available."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def print_system_info():
    """Print system information."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU: {platform.processor()}")
    print(
        f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )

    mem = psutil.virtual_memory()
    print(
        f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available"
    )

    gpu_mem = get_gpu_memory()
    if gpu_mem is not None:
        print(f"GPU Memory Used: {gpu_mem:.1f} MB")
    else:
        print("GPU: Not available or not NVIDIA")

    print("=" * 70)
    print()


def print_stats_summary(stats_list: list[BenchmarkStats]):
    """Print summary of benchmark statistics across different frame counts.

    Args:
        stats_list: List of BenchmarkStats for different frame counts
    """
    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS SUMMARY")
    print("=" * 80)

    for stats in stats_list:
        print(
            f"\n{stats.frame_count} Frame{'s' if stats.frame_count > 1 else ''} ({stats.iterations} iterations)"
        )
        print("-" * 80)

        # Timing statistics
        print(f"Query Time:")
        print(f"  Mean: {stats.mean_query_time:.3f}s ± {stats.std_query_time:.3f}s")
        print(f"  Range: [{stats.min_query_time:.3f}s - {stats.max_query_time:.3f}s]")
        stability = (
            (1 - stats.std_query_time / stats.mean_query_time) * 100
            if stats.mean_query_time > 0
            else 0
        )
        print(f"  Stability: {stability:.1f}% (lower std = more stable)")

        # Token statistics
        if stats.mean_prompt_tokens is not None:
            print(f"\nToken Usage (mean across {stats.iterations} runs):")
            print(f"  Prompt: {stats.mean_prompt_tokens:.1f} tokens")
            print(f"  Completion: {stats.mean_completion_tokens:.1f} tokens")
            print(f"  Total: {stats.mean_total_tokens:.1f} tokens")

        # Throughput statistics
        if stats.mean_tokens_per_second is not None:
            print(f"\nThroughput:")
            print(
                f"  Mean: {stats.mean_tokens_per_second:.2f} ± {stats.std_tokens_per_second:.2f} tokens/sec"
            )

    # Comparison table
    if len(stats_list) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON (Mean Values)")
        print("=" * 80)
        print(
            f"{'Frames':<8} {'Iterations':<12} {'Mean Time':<15} {'Std Time':<15} "
            f"{'Mean Tok/s':<15} {'Std Tok/s':<12}"
        )
        print("-" * 80)

        for stats in stats_list:
            mean_tps = (
                f"{stats.mean_tokens_per_second:.2f}"
                if stats.mean_tokens_per_second
                else "N/A"
            )
            std_tps = (
                f"{stats.std_tokens_per_second:.2f}"
                if stats.std_tokens_per_second
                else "N/A"
            )
            print(
                f"{stats.frame_count:<8} "
                f"{stats.iterations:<12} "
                f"{stats.mean_query_time:<15.3f} "
                f"{stats.std_query_time:<15.3f} "
                f"{mean_tps:<15} "
                f"{std_tps:<12}"
            )

        # Scaling analysis
        print("\n" + "=" * 80)
        print("SCALING ANALYSIS")
        print("=" * 80)

        base_stats = stats_list[0]
        for stats in stats_list[1:]:
            frame_increase = stats.frame_count / base_stats.frame_count
            time_increase = stats.mean_query_time / base_stats.mean_query_time

            print(f"\n{stats.frame_count}x frames vs {base_stats.frame_count}x frame:")
            print(f"  Frame increase: {frame_increase:.1f}x")
            print(
                f"  Time increase: {time_increase:.2f}x "
                f"({stats.mean_query_time:.3f}s vs {base_stats.mean_query_time:.3f}s)"
            )

            if stats.mean_prompt_tokens and base_stats.mean_prompt_tokens:
                token_increase = (
                    stats.mean_prompt_tokens / base_stats.mean_prompt_tokens
                )
                additional_tokens = (
                    stats.mean_prompt_tokens - base_stats.mean_prompt_tokens
                )
                tokens_per_frame = additional_tokens / (
                    stats.frame_count - base_stats.frame_count
                )
                print(f"  Token increase: {token_increase:.2f}x")
                print(
                    f"  Additional tokens: +{additional_tokens:.0f} (~{tokens_per_frame:.0f} per frame)"
                )

            # Calculate efficiency
            efficiency = frame_increase / time_increase if time_increase > 0 else 0
            print(
                f"  Scaling efficiency: {efficiency:.2f} (1.0 = linear, >1.0 = better than linear)"
            )

    print("=" * 80)


def print_metrics_summary(metrics_list: list[PerformanceMetrics]):
    """Print a summary of all performance metrics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 70)

    for i, m in enumerate(metrics_list, 1):
        print(f"\nTest {i}: {m.frame_count} frame{'s' if m.frame_count > 1 else ''}")
        print("-" * 70)
        print(f"Query: {m.query}")
        print(f"\nResponse ({m.response_length} chars):")
        print(f"  {m.response}")

        print(f"\nToken Usage:")
        if m.actual_prompt_tokens is not None:
            print(f"  ACTUAL (from vLLM API):")
            print(f"    - Prompt tokens: {m.actual_prompt_tokens}")
            print(f"    - Completion tokens: {m.actual_completion_tokens}")
            print(f"    - Total tokens: {m.actual_total_tokens}")

            # Calculate image vs text breakdown
            image_tokens = m.actual_prompt_tokens - m.estimated_prompt_tokens
            print(f"\n  Breakdown (calculated):")
            print(f"    - Text tokens: ~{m.estimated_prompt_tokens}")
            print(f"    - Image tokens: ~{image_tokens} ({m.frame_count} frames)")
            print(
                f"    - Image tokens per frame: ~{image_tokens // m.frame_count if m.frame_count > 0 else 0}"
            )

            if m.estimated_output_tokens:
                accuracy = (
                    m.actual_completion_tokens / m.estimated_output_tokens * 100
                    if m.estimated_output_tokens > 0
                    else 0
                )
                print(f"\n  Estimation accuracy:")
                print(
                    f"    - Estimated output: {m.estimated_output_tokens} tokens "
                    f"(actual: {m.actual_completion_tokens}, {accuracy:.0f}% accurate)"
                )
        else:
            print(f"  ESTIMATED (API usage not available):")
            print(f"    - Text prompt: ~{m.estimated_prompt_tokens} tokens")
            print(
                f"    - Images ({m.frame_count}x): ~{m.estimated_image_tokens} tokens"
            )
            print(f"    - Output: ~{m.estimated_output_tokens} tokens")

        print(f"\nTiming:")
        print(f"  Connection: {m.connect_time:.3f}s")
        print(f"  Query Processing: {m.query_time:.3f}s")
        print(f"  Total: {m.total_time:.3f}s")

        if m.tokens_per_second:
            print(f"  Throughput: {m.tokens_per_second:.2f} output tokens/sec")

        if m.memory_used_mb:
            print(f"\nMemory:")
            print(f"  System RAM: {m.memory_used_mb:.1f} MB")

        if m.gpu_memory_used_mb:
            print(f"  GPU VRAM: {m.gpu_memory_used_mb:.1f} MB")

    # Comparison
    if len(metrics_list) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        # Use actual tokens if available, otherwise estimated
        use_actual = metrics_list[0].actual_prompt_tokens is not None

        if use_actual:
            print(
                f"{'Frames':<8} {'Prompt':<10} {'Compl':<10} {'Total':<10} {'Query Time':<12} {'Tok/sec':<10}"
            )
            print("-" * 70)
            for m in metrics_list:
                tps = f"{m.tokens_per_second:.1f}" if m.tokens_per_second else "N/A"
                print(
                    f"{m.frame_count:<8} "
                    f"{m.actual_prompt_tokens:<10} "
                    f"{m.actual_completion_tokens:<10} "
                    f"{m.actual_total_tokens:<10} "
                    f"{m.query_time:<12.3f} "
                    f"{tps:<10}"
                )
        else:
            print(
                f"{'Frames':<8} {'Input~':<10} {'Output~':<10} {'Total~':<10} {'Query Time':<12} {'Tok/sec':<10}"
            )
            print("-" * 70)
            for m in metrics_list:
                total_est = (
                    (m.estimated_prompt_tokens or 0)
                    + (m.estimated_image_tokens or 0)
                    + (m.estimated_output_tokens or 0)
                )
                input_est = (m.estimated_prompt_tokens or 0) + (
                    m.estimated_image_tokens or 0
                )
                tps = f"{m.tokens_per_second:.1f}" if m.tokens_per_second else "N/A"
                print(
                    f"{m.frame_count:<8} "
                    f"{input_est:<10} "
                    f"{m.estimated_output_tokens or 0:<10} "
                    f"{total_est:<10} "
                    f"{m.query_time:<12.3f} "
                    f"{tps:<10}"
                )

        # Analysis
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        base_metrics = metrics_list[0]
        for m in metrics_list[1:]:
            frame_increase = m.frame_count / base_metrics.frame_count

            if use_actual:
                input_token_increase = (
                    m.actual_prompt_tokens / base_metrics.actual_prompt_tokens
                )
                print(f"\n{m.frame_count}x frames vs {base_metrics.frame_count}x:")
                print(f"  Frame count: {frame_increase:.1f}x")
                print(
                    f"  Prompt tokens: {input_token_increase:.2f}x "
                    f"({m.actual_prompt_tokens} vs {base_metrics.actual_prompt_tokens})"
                )
                print(
                    f"  Additional prompt tokens: +{m.actual_prompt_tokens - base_metrics.actual_prompt_tokens} "
                    f"(~{(m.actual_prompt_tokens - base_metrics.actual_prompt_tokens) // (m.frame_count - base_metrics.frame_count)} per additional frame)"
                )
            else:
                base_input = (base_metrics.estimated_prompt_tokens or 0) + (
                    base_metrics.estimated_image_tokens or 0
                )
                m_input = (m.estimated_prompt_tokens or 0) + (
                    m.estimated_image_tokens or 0
                )
                input_token_increase = m_input / base_input if base_input > 0 else 0
                print(f"\n{m.frame_count}x frames vs {base_metrics.frame_count}x:")
                print(f"  Frame count: {frame_increase:.1f}x")
                print(
                    f"  Input tokens: {input_token_increase:.2f}x ({m_input} vs {base_input})"
                )

            time_increase = m.query_time / base_metrics.query_time
            print(
                f"  Processing time: {time_increase:.2f}x ({m.query_time:.3f}s vs {base_metrics.query_time:.3f}s)"
            )
            print(f"  Time per frame: {m.query_time / m.frame_count:.3f}s")

    print("=" * 70)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Agent example using vLLM provider",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # vLLM server options
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="vLLM server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port",
    )

    # MQTT broker options
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="127.0.0.1",
        help="MQTT broker host address",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port",
    )

    # LLM generation options
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation (0.0-2.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (reduced for benchmarking)",
    )

    # Test image
    parser.add_argument(
        "--image",
        type=str,
        default="bird.jpg",
        help="Path to test image (relative to script or absolute)",
    )

    # Benchmark options
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per frame count for statistics",
    )

    return parser.parse_args()


async def run_benchmark(
    agent: VisualAgent,
    driver: MockWebRTCDriver,
    question: str,
    frame_count: int,
    system_prompt: str,
    image_size: tuple[int, int],
) -> PerformanceMetrics:
    """Run a single benchmark test with specified frame count.

    Args:
        agent: The VisualAgent instance
        driver: The mock WebRTC driver
        question: The question to ask
        frame_count: Number of frames to use (1, 3, 5, etc.)
        system_prompt: System prompt text for token counting
        image_size: (width, height) of the image for token estimation

    Returns:
        PerformanceMetrics with detailed performance data
    """
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {frame_count} frame{'s' if frame_count > 1 else ''}")
    print(f"{'=' * 70}")
    print(f"Question: {question}\n")

    # Calculate input token estimates
    system_prompt_tokens = estimate_text_tokens(system_prompt)
    user_query_tokens = estimate_text_tokens(question)
    prompt_tokens = system_prompt_tokens + user_query_tokens

    # Each frame is treated as an image
    image_tokens_per_frame = estimate_image_tokens(image_size[0], image_size[1])
    total_image_tokens = image_tokens_per_frame * frame_count

    total_input_tokens = prompt_tokens + total_image_tokens

    print(f"Input Token Estimate:")
    print(f"  System prompt: ~{system_prompt_tokens} tokens")
    print(f"  User query: ~{user_query_tokens} tokens")
    print(
        f"  Images ({frame_count}x): ~{total_image_tokens} tokens ({image_tokens_per_frame} per frame)"
    )
    print(f"  Total input: ~{total_input_tokens} tokens\n")

    # Record initial state
    start_total = time.time()
    mem_before = psutil.virtual_memory().used / (1024**2)
    gpu_mem_before = get_gpu_memory()

    # Update driver's buffer size
    driver.video_buffer_size = frame_count

    # Connect driver (loads frames)
    print(f"Loading {frame_count} frame(s)...")
    connect_start = time.time()
    await driver.connect()
    connect_time = time.time() - connect_start
    print(f"✓ Frames loaded in {connect_time:.3f}s\n")

    # Run query - we need to call the internal method to get full response with usage
    print("Processing query...")
    query_start = time.time()

    # Get frames for processing
    frames = await driver.get_frame_buffer()
    audio = await driver.get_audio_buffer()

    # Call _process_visual_input directly to get the full LLMChatResponse
    llm_response = await agent._process_visual_input(frames, audio, question)
    response = llm_response  # This is the text response

    query_time = time.time() - query_start
    total_time = time.time() - start_total

    # Measure memory usage
    mem_after = psutil.virtual_memory().used / (1024**2)
    gpu_mem_after = get_gpu_memory()

    mem_used = mem_after - mem_before
    gpu_mem_used = (
        (gpu_mem_after - gpu_mem_before) if gpu_mem_before and gpu_mem_after else None
    )

    # Try to get actual token usage from agent's last LLM call
    actual_prompt_tokens = None
    actual_completion_tokens = None
    actual_total_tokens = None

    # Check if agent has dialogue history with usage info
    if hasattr(agent, "dialogue_history") and agent.dialogue_history:
        # Get the last assistant message response object
        last_entry = agent.dialogue_history[-1]
        if "response" in last_entry and hasattr(last_entry["response"], "usage"):
            usage = last_entry["response"].usage
            if usage:
                actual_prompt_tokens = usage.get("prompt_tokens")
                actual_completion_tokens = usage.get("completion_tokens")
                actual_total_tokens = usage.get("total_tokens")

    # Calculate estimated output tokens (for comparison)
    estimated_output_tokens = estimate_text_tokens(response)
    tokens_per_second = (
        (actual_completion_tokens or estimated_output_tokens) / query_time
        if query_time > 0
        else None
    )

    # Compact output for iterations - full details shown in summary

    # Disconnect for clean next test
    await driver.disconnect()

    return PerformanceMetrics(
        frame_count=frame_count,
        query=question,
        response=response,
        response_length=len(response),
        total_time=total_time,
        connect_time=connect_time,
        query_time=query_time,
        actual_prompt_tokens=actual_prompt_tokens,
        actual_completion_tokens=actual_completion_tokens,
        actual_total_tokens=actual_total_tokens,
        estimated_prompt_tokens=prompt_tokens,
        estimated_image_tokens=total_image_tokens,
        estimated_output_tokens=estimated_output_tokens,
        tokens_per_second=tokens_per_second,
        memory_used_mb=mem_used,
        gpu_memory_used_mb=gpu_mem_used,
    )


async def main():
    args = parse_args()

    # Print system information first
    print_system_info()

    # Build configuration from arguments
    CONFIG = {
        "llm": {
            "provider": "vllm",
            # Model name will be auto-detected from the server
            "vllm_base_url": f"http://{args.host}:{args.port}/v1",
            # "vllm_model_name": "auto-detected",  # Optional: omit to auto-detect
            "vllm_api_key": "dummy",  # vLLM doesn't require real API key
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "messaging": {
            "broker_address": args.mqtt_host,
            "port": args.mqtt_port,
        },
    }

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"vLLM Server: {args.host}:{args.port}")
    print(f"MQTT Broker: {args.mqtt_host}:{args.mqtt_port}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Test Image: {args.image}")
    print("=" * 70)
    print()

    print("Setting up connections and agents...")
    print()

    # Connect to MQTT
    print("1. Connecting to MQTT broker...")
    messager = Messager(
        broker_address=CONFIG["messaging"]["broker_address"],
        port=CONFIG["messaging"]["port"],
    )
    await messager.connect()
    print("   ✓ Connected\n")

    # Create LLM client
    print("2. Connecting to vLLM server...")
    print(f"   Server: {CONFIG['llm']['vllm_base_url']}")
    print("   Model: auto-detected from server")
    llm = LLMFactory.create(CONFIG, messager)
    print("   ✓ LLM client ready\n")

    # Setup image paths - load all test images
    print("3. Loading test images...")
    test_images_dir = Path(__file__).parent / "test_images"

    # Collect all available test images
    image_paths = []
    if test_images_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(sorted(test_images_dir.glob(ext)))

    if not image_paths:
        print(f"   ✗ Error: No test images found in {test_images_dir}")
        print("   Please ensure test images exist in the test_images directory")
        await messager.disconnect()
        return

    print(f"   Found {len(image_paths)} test images:")
    for img_path in image_paths:
        img = Image.open(img_path)
        print(f"     - {img_path.name}: {img.size[0]}x{img.size[1]} {img.mode}")

    # Use first image for size reference
    reference_img = Image.open(image_paths[0])
    image_size = reference_img.size
    print(f"   Using reference size: {image_size[0]}x{image_size[1]}")
    print("   ✓ Images loaded\n")

    # Create driver (will be updated per test)
    driver = MockWebRTCDriver(image_paths, video_buffer_size=1)

    # Create agent
    print("4. Creating VisualAgent...")
    tool_manager = ToolManager(messager)
    await tool_manager.async_init()

    agent = VisualAgent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        webrtc_driver=driver,
        role="visual_assistant",
        system_prompt=(
            "You are a visual AI assistant. Analyze images and answer questions "
            "about them accurately and concisely."
        ),
        enable_dialogue_logging=True,
        enable_auto_processing=False,
    )
    await agent.async_init()
    print("   ✓ Agent ready\n")

    # Define test question and get system prompt
    question = "Describe the bird in this image. What species might it be?"
    system_prompt = (
        "You are a visual AI assistant. Analyze images and answer questions "
        "about them accurately and concisely."
    )

    # Run benchmarks with different frame counts
    print("=" * 80)
    print("RUNNING BENCHMARKS")
    print("=" * 80)
    print(f"Configuration: {args.iterations} iterations per frame count")
    print(f"Frame counts to test: 1, 3, 5")
    print(f"Total tests: {args.iterations * 3}")
    print("=" * 80)

    all_stats = []

    try:
        # Test with 1, 3, and 5 frames
        for frame_count in [1, 3, 5]:
            print(f"\n{'=' * 80}")
            print(
                f"TESTING {frame_count} FRAME{'S' if frame_count > 1 else ''} ({args.iterations} iterations)"
            )
            print(f"{'=' * 80}")

            iteration_metrics = []

            for iteration in range(args.iterations):
                print(f"\nIteration {iteration + 1}/{args.iterations}:")

                metrics = await run_benchmark(
                    agent, driver, question, frame_count, system_prompt, image_size
                )
                iteration_metrics.append(metrics)

                # Show quick summary
                if metrics.actual_prompt_tokens:
                    print(
                        f"  → {metrics.query_time:.3f}s, "
                        f"{metrics.actual_completion_tokens} tokens, "
                        f"{metrics.tokens_per_second:.2f} tok/s"
                    )
                else:
                    print(f"  → {metrics.query_time:.3f}s")

                # Small delay between iterations to avoid overwhelming the server
                if iteration < args.iterations - 1:
                    await asyncio.sleep(0.5)

            # Calculate statistics for this frame count
            stats = calculate_benchmark_stats(iteration_metrics)
            all_stats.append(stats)

            print(f"\n✓ Completed {frame_count} frame test:")
            print(
                f"  Mean time: {stats.mean_query_time:.3f}s ± {stats.std_query_time:.3f}s"
            )
            if stats.mean_tokens_per_second:
                print(
                    f"  Mean throughput: {stats.mean_tokens_per_second:.2f} ± {stats.std_tokens_per_second:.2f} tok/s"
                )

        # Print comprehensive statistical summary
        print_stats_summary(all_stats)

        # Optionally print detailed metrics for last iteration of last test
        if all_stats:
            print("\n" + "=" * 80)
            print("SAMPLE DETAILED METRICS (Last iteration of 5-frame test)")
            print("=" * 80)
            last_metric = all_stats[-1].all_metrics[-1]
            print(f"Query: {last_metric.query}")
            print(f"Response: {last_metric.response}")
            if last_metric.actual_prompt_tokens:
                print(f"\nTokens:")
                print(f"  Prompt: {last_metric.actual_prompt_tokens}")
                print(f"  Completion: {last_metric.actual_completion_tokens}")
                print(f"  Total: {last_metric.actual_total_tokens}")
            print(f"\nTiming:")
            print(f"  Query: {last_metric.query_time:.3f}s")
            print(f"  Throughput: {last_metric.tokens_per_second:.2f} tok/s")

    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\n" + "=" * 70)
        print("CLEANUP")
        print("=" * 70)
        await agent.stop()
        await driver.disconnect()
        await messager.disconnect()
        print("✓ All resources cleaned up\n")

    print("=" * 70)
    print("✨ BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
