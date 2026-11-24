"""Markov chain transition probability matrix estimation and matrix scaling."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from greenplan.config import ScenarioConfig


def compute_transition_matrix(
    initial_map: np.ndarray, final_map: np.ndarray, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical transition counts and stochastic transition probability matrix P.

    Returns:
        transition_counts: Matrix of pixel transition counts (k x k).
        transition_matrix: Row-stochastic probability matrix P where P[i, j] = P(j at t2 | i at t1).
                           Every row strictly sums to 1.0.
    """
    valid = (
        (initial_map >= 0)
        & (initial_map < num_classes)
        & (final_map >= 0)
        & (final_map < num_classes)
    )
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i, f in zip(initial_map[valid], final_map[valid]):
        counts[i, f] += 1

    prob_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    row_sums = np.sum(counts, axis=1)

    for i in range(num_classes):
        if row_sums[i] > 0:
            prob_matrix[i, :] = counts[i, :] / row_sums[i]
        else:
            prob_matrix[i, i] = 1.0  # Identity persistence for unobserved states

    return counts, prob_matrix


def scale_transition_matrix(prob_matrix: np.ndarray, power: int) -> np.ndarray:
    """Scale transition probability matrix across multiple time-steps using matrix exponentiation.

    P(t) = P^t
    """
    if power <= 0:
        raise ValueError(f"Matrix power must be >= 1, got {power}")
    scaled = np.linalg.matrix_power(prob_matrix, power)
    # Re-normalize to ensure strict row-stochasticity against numerical rounding
    row_sums = np.sum(scaled, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return scaled / row_sums


def project_target_areas(
    current_map: np.ndarray, prob_matrix: np.ndarray, num_classes: int
) -> np.ndarray:
    """Project future class pixel counts given current map and transition matrix."""
    current_counts = np.bincount(current_map.flatten(), minlength=num_classes)[:num_classes]
    # Projected area: N(t+1) = N(t) * P
    projected = np.dot(current_counts.astype(np.float64), prob_matrix)
    target_counts = np.round(projected).astype(np.int64)

    # Adjust rounding differences to strictly conserve total pixel area
    diff = int(np.sum(current_counts) - np.sum(target_counts))
    if diff != 0:
        # Allocate difference to the class with largest fractional part
        target_counts[np.argmax(projected - np.floor(projected))] += diff

    return target_counts


def load_scenarios_config(yaml_path: Path | str = "configs/scenarios.yaml") -> List[ScenarioConfig]:
    """Load simulation scenarios configuration from YAML.

    Defines scenario specifications (protection masks, green buffer distances).
    """
    p = Path(yaml_path)
    if not p.exists():
        return [
            ScenarioConfig(name="BAU", description="Business-As-Usual", protection_mask=False),
            ScenarioConfig(name="GreenProtection", description="Green Protection", protection_mask=True),
        ]
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [ScenarioConfig(**scen) for scen in data.get("scenarios", [])]
