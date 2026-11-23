"""Cellular Automata (CA) spatial allocation engine."""

from typing import Dict

import numpy as np
from scipy.ndimage import uniform_filter


def compute_neighborhood_effect(
    class_map: np.ndarray, target_class: int, kernel_size: int = 5
) -> np.ndarray:
    """Calculate spatial neighborhood contiguity (fraction of neighborhood belonging to target_class)."""
    binary_mask = (class_map == target_class).astype(np.float32)
    neighborhood = uniform_filter(binary_mask, size=kernel_size, mode="reflect")
    return neighborhood


def allocate_cellular_automata(
    initial_map: np.ndarray,
    target_counts: np.ndarray,
    suitability_maps: Dict[int, np.ndarray],
    constraint_mask: np.ndarray,
    kernel_size: int = 5,
    max_iterations: int = 15,
    random_seed: int = 42,
) -> np.ndarray:
    """Run Cellular Automata spatial allocation until Markov target areas are met.

    Parameters:
        initial_map: 2D array of initial LULC class indices (H x W).
        target_counts: 1D array of target pixel counts per class (length k).
        suitability_maps: Dict mapping class_id -> 2D suitability array (H x W, values in [0, 1]).
        constraint_mask: 2D boolean array (True where development is allowed, False where prohibited).
        kernel_size: Size of CA moving spatial neighborhood filter.
        max_iterations: Number of allocation adjustment cycles.
        random_seed: Seed for stochastic tie-breaking.

    Returns:
        simulated_map: 2D array of simulated future LULC classes with conserved total area.
    """
    rng = np.random.default_rng(random_seed)
    h, w = initial_map.shape
    num_classes = len(target_counts)
    simulated = initial_map.copy()

    # Priority order for urban growth simulation:
    # Allocate urban classes (e.g. Built-up) first, then green/farmland, leaving baseline/bare.
    for iteration in range(max_iterations):
        for class_id in range(num_classes):
            current_count = int(np.sum(simulated == class_id))
            target_count = int(target_counts[class_id])
            needed = target_count - current_count

            if needed <= 0:
                continue

            # Compute neighborhood effect
            neigh = compute_neighborhood_effect(simulated, class_id, kernel_size=kernel_size)
            suit = suitability_maps.get(class_id, np.ones((h, w), dtype=np.float32))

            # Potential = Suitability * Neighborhood * Constraint * Tiebreaker noise
            noise = rng.uniform(0.98, 1.02, size=(h, w)).astype(np.float32)
            potential = suit * (0.3 + 0.7 * neigh) * constraint_mask.astype(np.float32) * noise

            # Cannot convert already assigned pixels of this class or protected pixels
            candidate_mask = (simulated != class_id) & constraint_mask
            if not np.any(candidate_mask):
                continue

            candidate_indices = np.flatnonzero(candidate_mask)
            candidate_potentials = potential.ravel()[candidate_indices]

            # Pick top candidates
            k_alloc = min(needed, len(candidate_indices))
            if k_alloc > 0:
                top_indices = candidate_indices[
                    np.argpartition(candidate_potentials, -k_alloc)[-k_alloc:]
                ]
                simulated.ravel()[top_indices] = class_id

    return simulated
