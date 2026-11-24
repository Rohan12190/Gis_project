"""Unit tests for Cellular Automata allocation engine."""

import numpy as np

from greenplan.simulation.ca import allocate_cellular_automata


def test_ca_allocation_guarantees_target_counts():
    h, w = 40, 40
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


def test_ca_exact_area_allocation_small_grid():
    """Verify exact cell-for-cell target count matching on a small unconstrained grid."""
    h, w = 10, 10
    initial_map = np.full((h, w), 1, dtype=np.uint8)

    # Target: 30 cells class 0, 40 cells class 1, 30 cells class 2 = 100 cells
    target_counts = np.array([30, 40, 30])
    constraint_mask = np.ones((h, w), dtype=bool)

    rng = np.random.default_rng(123)
    suitability_maps = {
        0: rng.uniform(0.2, 0.8, size=(h, w)).astype(np.float32),
        1: rng.uniform(0.2, 0.8, size=(h, w)).astype(np.float32),
        2: rng.uniform(0.2, 0.8, size=(h, w)).astype(np.float32),
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
    assert np.array_equal(actual_counts, target_counts), (
        f"Expected exact counts {target_counts}, got {actual_counts}"
    )


def test_multi_seed_simulation_reports_mean_and_std():
    """Verify multi-seed simulation execution computes mean and standard deviation."""
    h, w = 15, 15
    initial_map = np.random.choice([0, 1, 2], size=(h, w), p=[0.2, 0.4, 0.4])
    ground_truth_t2 = np.random.choice([0, 1, 2], size=(h, w), p=[0.3, 0.3, 0.4])

    target_counts = np.bincount(ground_truth_t2.ravel(), minlength=3)
    suitability_maps = {
        0: np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32),
        1: np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32),
        2: np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32),
    }
    constraint_mask = np.ones((h, w), dtype=bool)

    seeds = [10, 20, 30, 40, 50]
    runs = []
    from greenplan.validation.metrics import compute_figure_of_merit

    for s in seeds:
        sim = allocate_cellular_automata(
            initial_map=initial_map,
            target_counts=target_counts,
            suitability_maps=suitability_maps,
            constraint_mask=constraint_mask,
            random_seed=s,
        )
        fom = compute_figure_of_merit(initial_map, ground_truth_t2, sim, target_class=0)
        runs.append(fom)

    foms = [r["figure_of_merit"] for r in runs]
    mean_fom = float(np.mean(foms))
    std_fom = float(np.std(foms))

    assert 0.0 <= mean_fom <= 1.0
    assert std_fom >= 0.0
