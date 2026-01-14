"""
training.py - Training example extracted from ai-performance-engineering benchmark.

Based on: https://github.com/cfregly/ai-performance-engineering/blob/main/code/ch01/optimized_performance.py

This demonstrates:
1. 3-layer MLP model with ReLU activations
2. FP16 precision for faster GPU training (tensor cores)
3. Batch fusion optimization
4. Proper warmup before timing
5. GPU timing with torch.cuda.Event
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---- 1. Model Definition ----
class TrainingModel(nn.Module):
    """
    3-layer MLP for classification.

    Architecture:
        Input (hidden_dim) → Linear → ReLU → Linear → ReLU → Linear → Output (10)

    This is deeper than inference.py's single layer, enabling:
    - Non-linear pattern learning (via ReLU activations)
    - Hierarchical feature extraction
    - Enough compute to saturate GPU cores
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Sequential container chains layers together
        # Data flows: input → layer1 → relu → layer2 → relu → layer3 → output
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Layer 1: hidden → hidden
            nn.ReLU(),  # Activation 1
            nn.Linear(hidden_dim, hidden_dim),  # Layer 2: hidden → hidden
            nn.ReLU(),  # Activation 2
            nn.Linear(hidden_dim, 10),  # Layer 3: hidden → 10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())


# ---- 2. Training Configuration ----
class TrainingConfig:
    """Training hyperparameters."""

    hidden_dim: int = 512
    batch_size: int = 32
    num_microbatches: int = 64  # Number of microbatches per iteration
    fusion_factor: int = 8  # Fuse N microbatches into one larger batch
    learning_rate: float = 1e-4  # Adam works better with lower LR than SGD
    num_iterations: int = 5
    warmup_iterations: int = 3
    use_fp16: bool = True  # Use FP16 for tensor core acceleration


# ---- 3. Data Generation ----
def generate_synthetic_data(
    config: TrainingConfig, device: torch.device, dtype: torch.dtype
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Generate synthetic training data.

    In real training, replace with DataLoader loading actual data.
    Synthetic data is useful for:
    - Benchmarking (consistent, reproducible)
    - Testing training pipeline
    - Isolating compute from I/O
    """
    microbatches = [
        torch.randn(
            config.batch_size, config.hidden_dim, device=device, dtype=dtype
        ).contiguous()  # contiguous() ensures optimal memory layout
        for _ in range(config.num_microbatches)
    ]

    targets = [
        torch.randint(
            0,
            10,  # Random class labels 0-9
            (config.batch_size,),
            device=device,
        )
        for _ in range(config.num_microbatches)
    ]

    return microbatches, targets


def fuse_batches(
    microbatches: list[torch.Tensor], targets: list[torch.Tensor], fusion_factor: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Fuse multiple small batches into fewer larger batches.

    Why batch fusion helps:
    - Fewer kernel launches (each launch has overhead)
    - Better GPU utilization (larger batches = more parallelism)
    - Amortizes fixed costs over more samples

    Example with fusion_factor=8:
        64 microbatches of size 32 → 8 fused batches of size 256
    """
    fused_batches = []
    fused_targets = []

    for start in range(0, len(microbatches), fusion_factor):
        batch = torch.cat(microbatches[start : start + fusion_factor], dim=0)
        target = torch.cat(targets[start : start + fusion_factor], dim=0)
        fused_batches.append(batch)
        fused_targets.append(target)

    return fused_batches, fused_targets


# ---- 4. Training Loop ----
def train_iteration(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    fused_batches: list[torch.Tensor],
    fused_targets: list[torch.Tensor],
) -> float:
    """
    Single training iteration over all fused batches.

    Training step for each batch:
    1. Zero gradients (clear from previous step)
    2. Forward pass (compute predictions)
    3. Compute loss (compare predictions to targets)
    4. Backward pass (compute gradients via backpropagation)
    5. Optimizer step (update weights using gradients)

    Returns:
        Average loss for this iteration
    """
    total_loss = 0.0

    for data, target in zip(fused_batches, fused_targets):
        # Zero gradients - set_to_none=True is faster than zero_grad()
        # Instead of setting gradients to zero tensors, sets them to None
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(data)

        # Compute cross-entropy loss for classification
        loss = nn.functional.cross_entropy(logits, target)

        # Backward pass - compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(fused_batches)


# ---- 5. GPU Timing Utilities ----
def time_training_iteration(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    fused_batches: list[torch.Tensor],
    fused_targets: list[torch.Tensor],
) -> tuple[float, float]:
    """
    Time a training iteration using CUDA events.

    Why CUDA events instead of time.time():
    - GPU operations are asynchronous
    - time.time() measures queue time, not execution time
    - CUDA events measure actual GPU execution

    Returns:
        (average_loss, elapsed_time_ms)
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    avg_loss = train_iteration(model, optimizer, fused_batches, fused_targets)
    end.record()

    # Wait for GPU to finish before reading elapsed time
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    return avg_loss, elapsed_ms


# ---- 6. Main Training Function ----
def run_training(config: TrainingConfig | None = None) -> dict[str, Any]:
    """
    Complete training run with timing and metrics.

    Returns:
        Dictionary with training metrics
    """
    if config is None:
        config = TrainingConfig()

    # Device and dtype setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.float16 if (config.use_fp16 and device.type == "cuda") else torch.float32
    )

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Precision: {'FP16' if dtype == torch.float16 else 'FP32'}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Microbatches: {config.num_microbatches}")
    print(f"Fusion factor: {config.fusion_factor}")
    print(f"Effective batch per fused: {config.batch_size * config.fusion_factor}")

    # Reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # Create model
    model = TrainingModel(hidden_dim=config.hidden_dim)

    if dtype == torch.float16:
        model = model.half()  # Convert to FP16 for tensor core acceleration

    model = model.to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Generate data
    microbatches, targets = generate_synthetic_data(config, device, dtype)
    fused_batches, fused_targets = fuse_batches(
        microbatches, targets, config.fusion_factor
    )

    print(f"Fused batches: {len(fused_batches)} x {fused_batches[0].shape}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Warmup - critical for accurate benchmarking
    # First iterations are slow due to:
    # - CUDA kernel compilation
    # - Memory allocation
    # - cuDNN autotuning
    print(f"\nWarming up ({config.warmup_iterations} iterations)...")
    for _ in range(config.warmup_iterations):
        _ = train_iteration(model, optimizer, fused_batches, fused_targets)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed training
    print(f"\nTraining ({config.num_iterations} iterations)...")
    print("-" * 60)

    times_ms = []
    losses = []
    samples_per_iteration = config.batch_size * config.num_microbatches

    for i in range(config.num_iterations):
        if device.type == "cuda":
            avg_loss, elapsed_ms = time_training_iteration(
                model, optimizer, fused_batches, fused_targets
            )
            throughput = samples_per_iteration / (elapsed_ms / 1000)
            print(
                f"  Iter {i + 1}: loss={avg_loss:.4f}, time={elapsed_ms:.2f}ms, "
                + f"throughput={throughput:.0f} samples/sec"
            )
            times_ms.append(elapsed_ms)
        else:
            import time

            start = time.perf_counter()
            avg_loss = train_iteration(model, optimizer, fused_batches, fused_targets)
            elapsed_ms = (time.perf_counter() - start) * 1000
            throughput = samples_per_iteration / (elapsed_ms / 1000)
            print(
                f"  Iter {i + 1}: loss={avg_loss:.4f}, time={elapsed_ms:.2f}ms, "
                + f"throughput={throughput:.0f} samples/sec"
            )
            times_ms.append(elapsed_ms)

        losses.append(avg_loss)

    # Summary
    avg_time_ms = sum(times_ms) / len(times_ms)
    avg_throughput = samples_per_iteration / (avg_time_ms / 1000)

    print("-" * 60)
    print(f"Average time: {avg_time_ms:.2f} ms")
    print(f"Average throughput: {avg_throughput:.0f} samples/sec")
    print(f"Final loss: {losses[-1]:.4f}")
    print("=" * 60)

    # Save trained weights (like inference.py expects)
    torch.save(model.state_dict(), "model_weights.pth")
    print("✓ Saved model weights to model_weights.pth")

    return {
        "avg_time_ms": avg_time_ms,
        "avg_throughput": avg_throughput,
        "final_loss": losses[-1],
        "times_ms": times_ms,
        "losses": losses,
    }


if __name__ == "__main__":
    _ = run_training()
