"""Top-1 Mixture-of-Experts layer and capacity benchmark.

The benchmark measures how expert capacity affects token drop rate and forward
latency. Run this module directly to generate the benchmark plots.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_MODEL_DIM = 16
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EXPERT_COUNT = 8
DEFAULT_TOKEN_COUNT = 10_000
DEFAULT_WARMUP_STEPS = 10
DEFAULT_BENCHMARK_STEPS = 50
DEFAULT_CAPACITY_FACTORS = tuple(factor * 0.25 for factor in range(2, 9))


class SwiGLU(nn.Module):
    """SwiGLU activation for a tensor containing gate and value projections."""

    def forward(self, inputs: Tensor) -> Tensor:
        gate_values, projected_values = inputs.chunk(2, dim=-1)
        return F.silu(gate_values) * projected_values


class Expert(nn.Module):
    """Feed-forward expert using a SwiGLU activation."""

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if model_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model_dim and hidden_dim must be positive")

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, hidden_dim * 2),
            SwiGLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)


class Router(nn.Module):
    """Produce a probability distribution over experts for each token."""

    def __init__(self, model_dim: int, expert_count: int) -> None:
        super().__init__()
        if model_dim <= 0 or expert_count <= 0:
            raise ValueError("model_dim and expert_count must be positive")

        self.gate = nn.Linear(model_dim, expert_count)

    def forward(self, inputs: Tensor) -> Tensor:
        routing_logits = self.gate(inputs)
        return torch.softmax(routing_logits, dim=-1)


class MoE(nn.Module):
    """Top-1 Mixture-of-Experts layer with a fixed per-expert capacity."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        expert_count: int,
        capacity: int,
    ) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.router = Router(model_dim, expert_count)
        self.experts = nn.ModuleList(
            Expert(model_dim, hidden_dim) for _ in range(expert_count)
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, list[int], list[int], int]:
        """Route tokens and return outputs plus expert-load statistics.

        Returns:
            A tuple containing the output tensor, loads before capacity limiting,
            loads after capacity limiting, and the total number of dropped tokens.
        """
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must have shape [token_count, model_dim], got {inputs.shape}"
            )

        routing_probabilities = self.router(inputs)
        selected_experts = routing_probabilities.argmax(dim=-1)
        outputs = torch.zeros_like(inputs)

        loads_before_capacity: list[int] = []
        loads_after_capacity: list[int] = []
        dropped_token_count = 0

        for expert_index, expert in enumerate(self.experts):
            expert_mask = selected_experts == expert_index
            selected_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            assigned_token_count = selected_indices.numel()
            loads_before_capacity.append(assigned_token_count)

            if assigned_token_count > self.capacity:
                dropped_token_count += assigned_token_count - self.capacity
                random_order = torch.randperm(
                    assigned_token_count,
                    device=selected_indices.device,
                )
                selected_indices = selected_indices[random_order[: self.capacity]]

            processed_token_count = selected_indices.numel()
            loads_after_capacity.append(processed_token_count)
            if processed_token_count == 0:
                continue

            expert_inputs = inputs[selected_indices]
            routing_weights = routing_probabilities[
                selected_indices, expert_index
            ].unsqueeze(-1)
            outputs[selected_indices] = routing_weights * expert(expert_inputs)

        return (
            outputs,
            loads_before_capacity,
            loads_after_capacity,
            dropped_token_count,
        )


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before recording a timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_capacity_factors(
    model: MoE,
    inputs: Tensor,
    capacity_factors: Sequence[float],
    *,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    benchmark_steps: int = DEFAULT_BENCHMARK_STEPS,
) -> tuple[list[float], list[float]]:
    """Benchmark drop rates and average latency for each capacity factor."""
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if benchmark_steps <= 0:
        raise ValueError("benchmark_steps must be positive")

    token_count = inputs.shape[0]
    expert_count = len(model.experts)
    drop_rates: list[float] = []
    average_latencies: list[float] = []

    model.eval()
    with torch.inference_mode():
        for capacity_factor in capacity_factors:
            if capacity_factor <= 0:
                raise ValueError("capacity factors must be positive")

            model.capacity = math.ceil(
                capacity_factor * token_count / expert_count
            )

            for _ in range(warmup_steps):
                model(inputs)

            synchronize_device(inputs.device)
            start_time = time.perf_counter()

            dropped_token_count = 0
            for _ in range(benchmark_steps):
                _, _, _, dropped_token_count = model(inputs)

            synchronize_device(inputs.device)
            elapsed_time = time.perf_counter() - start_time

            average_latencies.append(elapsed_time / benchmark_steps)
            drop_rates.append(dropped_token_count / token_count)

    return drop_rates, average_latencies


def save_metric_plot(
    capacity_factors: Sequence[float],
    values: Sequence[float],
    *,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Create and save one capacity-factor metric plot."""
    figure, axes = plt.subplots()
    axes.plot(capacity_factors, values, marker="o")
    axes.set_xlabel("Capacity Factor")
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.grid()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    """Run the default capacity benchmark and save its plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(DEFAULT_TOKEN_COUNT, DEFAULT_MODEL_DIM, device=device)
    model = MoE(
        model_dim=DEFAULT_MODEL_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        expert_count=DEFAULT_EXPERT_COUNT,
        capacity=1,
    ).to(device)

    drop_rates, average_latencies = benchmark_capacity_factors(
        model,
        inputs,
        DEFAULT_CAPACITY_FACTORS,
    )

    output_directory = Path(__file__).resolve().parent
    save_metric_plot(
        DEFAULT_CAPACITY_FACTORS,
        drop_rates,
        y_label="Drop Rate",
        title="Capacity Factor vs. Drop Rate",
        output_path=output_directory / "Drop_Rate.png",
    )
    save_metric_plot(
        DEFAULT_CAPACITY_FACTORS,
        average_latencies,
        y_label="Average Forward Time (seconds)",
        title="Capacity Factor vs. MoE Forward Time",
        output_path=output_directory / "Time.png",
    )


if __name__ == "__main__":
    main()
