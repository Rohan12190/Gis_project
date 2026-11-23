"""Unit tests for Cellular Automata allocation engine."""

import numpy as np

from greenplan.simulation.ca import allocate_cellular_automata


def test_ca_allocation_guarantees_target_counts():
    h, w = 40, 40
    total_pixels = h * w
    initial_map = np.random.choice([0, 1, 2], size=(h, w), p=[0.2, 0.3, 0.5])

    # Target counts strictly specified
    target_counts = np.array([500, 400, 700])  # sums to 1600
    constraint_mask = np.ones((h, w), dtype=bool)

    suitability_maps = {
        0: np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32),
        1: np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32),
        2: np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32),
    }

    simulated = allocate_cellular_automata(
        initial_map=initial_map,
        target_counts=target_counts,
        suitability_maps=suitability_maps,
        constraint_mask=constraint_mask,
        max_iterations=10,
        random_seed=42,
    )

    actual_counts = np.bincount(simulated.ravel(), minlength=3)
    np.testing.assert_allclose(actual_counts, target_counts, atol=2)
