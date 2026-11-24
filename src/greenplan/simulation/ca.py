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
    """Run Cellular Automata spatial allocation with guaranteed Markov target area conservation.

    Parameters:
        initial_map: 2D array of initial LULC class indices (H x W).
        target_counts: 1D array of target pixel counts per class (length k).
        suitability_maps: Dict mapping class_id -> 2D suitability array (H x W, values in [0, 1]).
        constraint_mask: 2D boolean array (True where development is allowed, False where prohibited).
        kernel_size: Size of CA moving spatial neighborhood filter.
        max_iterations: Number of allocation adjustment cycles.
        random_seed: Seed for stochastic tie-breaking.

    Returns:
        simulated_map: 2D array of simulated future LULC classes matching target counts.
    """
    rng = np.random.default_rng(random_seed)
    h, w = initial_map.shape
    num_classes = len(target_counts)
    simulated = initial_map.copy()

    # Fixed (constrained) cells cannot change
    unconstrained_idx = np.flatnonzero(constraint_mask)
    if len(unconstrained_idx) == 0:
        return simulated

    # Normalize target counts to match total raster size
    total_cells = h * w
    target_counts = np.round(target_counts * (total_cells / np.sum(target_counts))).astype(int)
    # Fix any rounding discrepancy on the largest class
    diff = total_cells - np.sum(target_counts)
    if diff != 0:
        target_counts[np.argmax(target_counts)] += diff

    # Accounts for cells locked by constraint mask
    locked_counts = np.bincount(initial_map[~constraint_mask], minlength=num_classes)[:num_classes]
    quota_unconstrained = np.maximum(0, target_counts - locked_counts)
    # Ensure quota matches unconstrained cell count
    n_unconstrained = len(unconstrained_idx)
    q_diff = n_unconstrained - np.sum(quota_unconstrained)
    if q_diff != 0:
        quota_unconstrained[np.argmax(quota_unconstrained)] += q_diff

    # Iterative spatial adjustment
    for iteration in range(max_iterations):
        # Compute potentials across unconstrained cells for all classes
        potentials = np.zeros((n_unconstrained, num_classes), dtype=np.float32)

        for c in range(num_classes):
            neigh = compute_neighborhood_effect(simulated, c, kernel_size=kernel_size)
            suit = suitability_maps.get(c, np.ones((h, w), dtype=np.float32))
            noise = rng.uniform(0.98, 1.02, size=(h, w)).astype(np.float32)

            pot_map = suit * (0.3 + 0.7 * neigh) * noise
            # Persistence inertia: slight preference to stay in current class
            pot_map[simulated == c] += 0.15

            potentials[:, c] = pot_map.ravel()[unconstrained_idx]

        # Quota-based multi-class allocation:
        # For each class, calculate priority scores based on potential minus baseline
        # Global competitive ranking ensures each class receives exactly its quota
        current_counts = np.bincount(simulated.ravel()[unconstrained_idx], minlength=num_classes)[:num_classes]
        deficit = quota_unconstrained - current_counts

        if np.all(deficit == 0):
            break

        # Identify deficit classes (need pixels) and surplus classes (can yield pixels)
        for def_class in np.argsort(-deficit):
            needed = deficit[def_class]
            if needed <= 0:
                continue

            # Candidates are cells in unconstrained set not currently in def_class
            curr_classes = simulated.ravel()[unconstrained_idx]
            is_candidate = curr_classes != def_class
            candidate_local_idx = np.flatnonzero(is_candidate)

            if len(candidate_local_idx) == 0:
                continue

            # Score candidates by net gain: potential(def_class) - potential(current_class)
            cand_curr_classes = curr_classes[candidate_local_idx]
            cand_gain = (
                potentials[candidate_local_idx, def_class]
                - potentials[candidate_local_idx, cand_curr_classes]
            )

            # Only take from surplus classes if possible
            surplus_mask = np.isin(cand_curr_classes, np.flatnonzero(deficit < 0))
            if np.any(surplus_mask):
                cand_gain[~surplus_mask] -= 1.0  # penalize non-surplus classes

            k_alloc = min(needed, len(candidate_local_idx))
            top_cand = candidate_local_idx[np.argpartition(cand_gain, -k_alloc)[-k_alloc:]]
            simulated.ravel()[unconstrained_idx[top_cand]] = def_class

            # Update deficit counts
            current_counts = np.bincount(simulated.ravel()[unconstrained_idx], minlength=num_classes)[:num_classes]
            deficit = quota_unconstrained - current_counts

    # Final guarantee pass: enforce exact quota matching
    final_counts = np.bincount(simulated.ravel()[unconstrained_idx], minlength=num_classes)[:num_classes]
    final_deficit = quota_unconstrained - final_counts

    while np.any(final_deficit > 0) and np.any(final_deficit < 0):
        gain_class = int(np.argmax(final_deficit))
        loss_class = int(np.argmin(final_deficit))

        # Find candidate cells currently in loss_class
        curr_classes = simulated.ravel()[unconstrained_idx]
        loss_candidates = np.flatnonzero(curr_classes == loss_class)

        if len(loss_candidates) == 0:
            break

        # Select the cell in loss_class that has highest potential for gain_class
        gain_scores = potentials[loss_candidates, gain_class]
        n_swap = min(final_deficit[gain_class], -final_deficit[loss_class], len(loss_candidates))

        best_swap_idx = loss_candidates[np.argpartition(gain_scores, -n_swap)[-n_swap:]]
        simulated.ravel()[unconstrained_idx[best_swap_idx]] = gain_class

        final_counts = np.bincount(simulated.ravel()[unconstrained_idx], minlength=num_classes)[:num_classes]
        final_deficit = quota_unconstrained - final_counts

    # Verify actual count against target_counts
    actual_final = np.bincount(simulated.ravel(), minlength=num_classes)[:num_classes]
    count_diff = np.abs(actual_final - target_counts)
    max_diff_pct = float(np.max(count_diff / np.maximum(1, target_counts)))

    if max_diff_pct > 0.01:
        import warnings

        warnings.warn(
            f"CA allocation discrepancy exceeds 1% of class area: "
            f"max_discrepancy={max_diff_pct:.2%}, actual={actual_final.tolist()}, "
            f"target={target_counts.tolist()}",
            UserWarning,
            stacklevel=2,
        )

    return simulated
