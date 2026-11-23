"""Bivariate 3x3 planning matrix combining projected loss risk and thermal stress."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

DECISION_MATRIX = {
    ("high", "high"): "Protect & Infill Canopy (Urgent Priority)",
    ("high", "medium"): "Designate Green Buffer",
    ("high", "low"): "Preventive Green Corridor Protection",
    ("medium", "high"): "Thermal Infill & Cool Pavements",
    ("medium", "medium"): "Sustainable Urban Landscaping",
    ("medium", "low"): "Conservation Monitoring",
    ("low", "high"): "Urban Heat Island Mitigation & Pocket Parks",
    ("low", "medium"): "General Canopy Maintenance",
    ("low", "low"): "Baseline Ecological Preservation",
}


def classify_terciles(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Classify 2D array into 0 (low), 1 (medium), 2 (high) using terciles (33.3% and 66.7% percentiles)."""
    valid_vals = arr[mask]
    if len(valid_vals) == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    t1, t2 = np.percentile(valid_vals, [33.33, 66.67])
    out = np.zeros_like(arr, dtype=np.uint8)
    out[mask & (arr >= t1)] = 1
    out[mask & (arr >= t2)] = 2
    return out


def classify_quartiles(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Classify 2D array into low (<25%), medium (25-75%), high (>75%) for sensitivity analysis."""
    valid_vals = arr[mask]
    if len(valid_vals) == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    q1, q3 = np.percentile(valid_vals, [25.0, 75.0])
    out = np.zeros_like(arr, dtype=np.uint8)
    out[mask & (arr >= q1)] = 1
    out[mask & (arr >= q3)] = 2
    return out


def evaluate_planning_matrix(
    loss_risk: np.ndarray, thermal_stress: np.ndarray, mask: np.ndarray, use_quartiles: bool = False
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute 2D priority categorization and category area breakdown."""
    classify_fn = classify_quartiles if use_quartiles else classify_terciles

    risk_cat = classify_fn(loss_risk, mask)
    heat_cat = classify_fn(thermal_stress, mask)

    h, w = loss_risk.shape
    priority_map = np.zeros((h, w), dtype=np.uint8)

    # 1 to 9 priority indices
    priority_keys = list(DECISION_MATRIX.keys())
    key_to_id = {k: idx + 1 for idx, k in enumerate(priority_keys)}

    for (r_lbl, h_lbl), idx in key_to_id.items():
        r_num = {"low": 0, "medium": 1, "high": 2}[r_lbl]
        h_num = {"low": 0, "medium": 1, "high": 2}[h_lbl]
        priority_map[mask & (risk_cat == r_num) & (heat_cat == h_num)] = idx

    # Calculate area fractions
    total_valid = int(np.sum(mask))
    area_breakdown = {}
    for (r_lbl, h_lbl), idx in key_to_id.items():
        title = DECISION_MATRIX[(r_lbl, h_lbl)]
        count = int(np.sum(priority_map == idx))
        area_breakdown[title] = float(count / total_valid * 100.0) if total_valid > 0 else 0.0

    return priority_map, area_breakdown


def load_sourced_recommendations(yaml_path: Path | str) -> List[Dict]:
    """Load policy recommendation rulebook and verify citation source or assumption flag."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rules = data.get("recommendations", [])
    for r in rules:
        if not r.get("source") and not r.get("assumption"):
            raise ValueError(
                f"Rule '{r.get('id')}' must have either a 'source' citation or 'assumption: true'."
            )
    return rules
