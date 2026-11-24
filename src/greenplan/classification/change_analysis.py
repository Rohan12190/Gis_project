"""Multi-temporal LULC post-classification change detection and sanity auditing."""

from typing import Dict

import numpy as np

# Unrealistic transitions (from_class, to_class)
# Class 0: Built-up, 1: Green space, 2: Farmland, 3: Water, 4: Bare land
IMPLAUSIBLE_TRANSITIONS = [
    (0, 1, "Built-up to Green Space (Urban Reversal)"),
    (0, 2, "Built-up to Farmland (Agricultural De-urbanization)"),
    (0, 3, "Built-up to Water (Inundation/Pavement Conversion)"),
]


def compute_transition_stats(
    map_t1: np.ndarray, map_t2: np.ndarray, num_classes: int, pixel_area_ha: float = 0.09
) -> Dict:
    """Calculate transition matrix, net and gross change per class in hectares."""
    valid = (map_t1 >= 0) & (map_t1 < num_classes) & (map_t2 >= 0) & (map_t2 < num_classes)
    t1_valid = map_t1[valid]
    t2_valid = map_t2[valid]

    flat_idx = t1_valid * num_classes + t2_valid
    matrix_counts = np.bincount(flat_idx, minlength=num_classes**2).reshape((num_classes, num_classes))

    t1_totals = np.sum(matrix_counts, axis=1)
    t2_totals = np.sum(matrix_counts, axis=0)

    net_change_pixels = t2_totals - t1_totals
    net_change_ha = net_change_pixels * pixel_area_ha

    # Check implausible transitions
    sanity_audit = []
    total_valid_pixels = len(t1_valid)
    for from_c, to_c, desc in IMPLAUSIBLE_TRANSITIONS:
        count = int(matrix_counts[from_c, to_c])
        pct = float(count / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
        sanity_audit.append(
            {
                "transition": desc,
                "from_class": from_c,
                "to_class": to_c,
                "pixel_count": count,
                "area_ha": round(count * pixel_area_ha, 2),
                "percentage_of_landscape": round(pct, 3),
            }
        )

    return {
        "transition_matrix_counts": matrix_counts,
        "net_change_ha": net_change_ha.tolist(),
        "area_t1_ha": (t1_totals * pixel_area_ha).tolist(),
        "area_t2_ha": (t2_totals * pixel_area_ha).tolist(),
        "sanity_audit": sanity_audit,
    }
