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

# Named baseline risk weights for scenario-conditional land conversion indicators.
# NOTE: loss_risk is a scenario-conditional spatial indicator derived from the CA-Markov
# simulation output and suitability surface, not an unconstrained empirical transition probability.
BASE_RISK_GREEN_CONVERTED: float = 0.90
BASE_RISK_FARMLAND_CONVERTED: float = 0.65
BASE_RISK_HIGH_SUITABILITY: float = 0.75
SUITABILITY_RISK_WEIGHT: float = 0.40
SCENARIO_CONVERSION_WEIGHT: float = 0.60


def compute_loss_risk(
    current_lulc: np.ndarray,
    simulated_future_lulc: np.ndarray,
    urban_suitability: np.ndarray,
    green_class: int = 1,
    farmland_class: int = 2,
    builtup_class: int = 0,
) -> np.ndarray:
    """Compute continuous scenario-conditional loss risk surface.

    Combines simulated conversion under BAU with underlying urban conversion suitability.
    For existing green and agricultural cells, risk is a continuous blend of suitability
    and simulated loss, using element-wise maximum across components so high-risk cells
    are never demoted.
    """
    h, w = current_lulc.shape
    is_green = (current_lulc == green_class)
    is_farm = (current_lulc == farmland_class)

    green_lost = is_green & (simulated_future_lulc != green_class)
    farm_lost = is_farm & (simulated_future_lulc == builtup_class)

    # Component 1: Simulated conversion indicator under BAU scenario
    risk_sim = np.zeros((h, w), dtype=np.float32)
    risk_sim[green_lost] = BASE_RISK_GREEN_CONVERTED
    risk_sim[farm_lost] = BASE_RISK_FARMLAND_CONVERTED

    # Component 2: Continuous suitability-driven threat for vulnerable classes
    risk_suit = np.zeros((h, w), dtype=np.float32)
    suit_clipped = np.clip(urban_suitability, 0.0, 1.0)
    risk_suit[is_green] = suit_clipped[is_green] * BASE_RISK_HIGH_SUITABILITY
    risk_suit[is_farm] = suit_clipped[is_farm] * (BASE_RISK_FARMLAND_CONVERTED * 0.90)

    # Component 3: Continuous blended score for converted/threatened cells
    risk_blend = np.zeros((h, w), dtype=np.float32)
    active = is_green | is_farm
    risk_blend[active] = (
        SCENARIO_CONVERSION_WEIGHT * risk_sim[active]
        + SUITABILITY_RISK_WEIGHT * risk_suit[active]
    )

    # Combine with element-wise maximum so converted high-suitability cells get max score
    risk = np.maximum(risk_sim, risk_suit)
    risk = np.maximum(risk, risk_blend)

    return risk.astype(np.float32)


def compute_thermal_stress(
    lst: np.ndarray,
    valid_mask: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Normalize Land Surface Temperature into [0.0, 1.0] using robust percentiles on valid pixels."""
    valid_lst = lst[valid_mask]
    if len(valid_lst) == 0:
        return np.zeros_like(lst, dtype=np.float32)

    t_min, t_max = np.percentile(valid_lst, [p_low, p_high])
    if t_max <= t_min:
        return np.zeros_like(lst, dtype=np.float32)

    norm = (lst - t_min) / (t_max - t_min)
    thermal_stress = np.where(valid_mask, np.clip(norm, 0.0, 1.0), 0.0)
    return thermal_stress.astype(np.float32)


def classify_terciles(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Classify 2D array into 0 (low), 1 (medium), 2 (high).

    Safely handles zero-inflated arrays (such as loss risk surfaces where >50% cells are 0)
    to prevent percentile collapse/degeneration into all-'high'.
    """
    valid_vals = arr[mask]
    if len(valid_vals) == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    out = np.zeros_like(arr, dtype=np.uint8)

    # Check for zero-inflation (common in risk / loss surfaces)
    zero_fraction = np.mean(valid_vals <= 1e-6)
    if zero_fraction > 0.40:
        # Values <= 0 remain 0 (low risk / baseline)
        pos_vals = valid_vals[valid_vals > 1e-6]
        if len(pos_vals) > 0:
            p33, p67 = np.percentile(pos_vals, [33.33, 66.67])
            if p33 < p67:
                out[mask & (arr > 1e-6) & (arr < p67)] = 1  # Medium
                out[mask & (arr >= p67)] = 2  # High
            else:
                # If positive values have identical/low variance, use median split
                med = np.median(pos_vals)
                out[mask & (arr > 1e-6) & (arr <= med)] = 1
                out[mask & (arr > med)] = 2
        return out

    # Standard continuous distribution: compute 33.33% and 66.67% terciles
    t1, t2 = np.percentile(valid_vals, [33.33, 66.67])
    if t1 == t2:
        # Uniform or low-variance fallback
        med = np.median(valid_vals)
        out[mask & (arr > med)] = 2
        out[mask & (arr <= med)] = 0
    else:
        out[mask & (arr >= t1) & (arr < t2)] = 1
        out[mask & (arr >= t2)] = 2

    return out


def classify_quartiles(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Classify 2D array into low (<25%), medium (25-75%), high (>75%) for sensitivity analysis."""
    valid_vals = arr[mask]
    if len(valid_vals) == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    out = np.zeros_like(arr, dtype=np.uint8)
    zero_fraction = np.mean(valid_vals <= 1e-6)
    if zero_fraction > 0.40:
        pos_vals = valid_vals[valid_vals > 1e-6]
        if len(pos_vals) > 0:
            med = np.median(pos_vals)
            out[mask & (arr > 1e-6) & (arr < med)] = 1
            out[mask & (arr >= med)] = 2
        return out

    q1, q3 = np.percentile(valid_vals, [25.0, 75.0])
    if q1 == q3:
        med = np.median(valid_vals)
        out[mask & (arr > med)] = 2
    else:
        out[mask & (arr >= q1) & (arr < q3)] = 1
        out[mask & (arr >= q3)] = 2
    return out


def evaluate_planning_matrix(
    loss_risk: np.ndarray, thermal_stress: np.ndarray, mask: np.ndarray, use_quartiles: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute 2D priority categorization and category area breakdown.

    Returns:
        priority_map: (H, W) array with values 1..9 corresponding to DECISION_MATRIX.
        risk_cat: (H, W) array with 0 (low), 1 (medium), 2 (high).
        heat_cat: (H, W) array with 0 (low), 1 (medium), 2 (high).
        area_breakdown: Dict of category names and percentage cover.
    """
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

    return priority_map, risk_cat, heat_cat, area_breakdown


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


def evaluate_planning_units(
    priority_map: np.ndarray,
    risk_cat: np.ndarray,
    heat_cat: np.ndarray,
    mask: np.ndarray,
    recommendations: List[Dict],
    grid_size_cells: int = 4,
) -> List[Dict]:
    """Aggregate cell-level planning into planning unit blocks and evaluate recommendation triggers."""
    h, w = priority_map.shape
    units = []
    level_names = {0: "low", 1: "medium", 2: "high"}

    unit_idx = 0
    for r in range(0, h, grid_size_cells):
        for c in range(0, w, grid_size_cells):
            sub_mask = mask[r : min(r + grid_size_cells, h), c : min(c + grid_size_cells, w)]
            if not np.any(sub_mask):
                continue

            sub_risk = risk_cat[r : min(r + grid_size_cells, h), c : min(c + grid_size_cells, w)][sub_mask]
            sub_heat = heat_cat[r : min(r + grid_size_cells, h), c : min(c + grid_size_cells, w)][sub_mask]
            sub_pri = priority_map[r : min(r + grid_size_cells, h), c : min(c + grid_size_cells, w)][sub_mask]

            # Dominant categories in this planning unit
            maj_risk_val = int(np.bincount(sub_risk).argmax())
            maj_heat_val = int(np.bincount(sub_heat).argmax())
            maj_pri_val = int(np.bincount(sub_pri).argmax())

            risk_str = level_names[maj_risk_val]
            heat_str = level_names[maj_heat_val]

            # Find matching category name from priority
            priority_keys = list(DECISION_MATRIX.keys())
            pri_title = DECISION_MATRIX[priority_keys[maj_pri_val - 1]] if 1 <= maj_pri_val <= 9 else "Unclassified"

            # Evaluate recommendations trigger condition
            matched_action_ids = []
            env = {"risk": risk_str, "heat": heat_str}
            for rec in recommendations:
                cond = rec.get("trigger_condition", "")
                try:
                    # Safely evaluate condition string in restricted env
                    if eval(cond, {"__builtins__": {}}, env):
                        matched_action_ids.append(rec.get("id"))
                except Exception:
                    pass

            units.append(
                {
                    "unit_id": f"PU_{unit_idx:04d}",
                    "row": r,
                    "col": c,
                    "loss_risk_level": risk_str,
                    "thermal_stress_level": heat_str,
                    "priority_category": pri_title,
                    "recommended_action_ids": matched_action_ids,
                }
            )
            unit_idx += 1

    return units
