"""MCDM Methods Calculator — six methods + AHP weight derivation."""

import pandas as pd
import numpy as np
from sqlalchemy import text

# Saaty Random Consistency Index table
RI_VALUES = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
             6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_normalize(matrix: pd.DataFrame, criteria_types: np.ndarray) -> pd.DataFrame:
    """Linear normalization: benefit → x/max, cost → min/x."""
    norm = matrix.copy().astype(float)
    for j, crit_type in enumerate(criteria_types):
        col = matrix.iloc[:, j].astype(float)
        if crit_type == 'benefit':
            max_val = col.max()
            norm.iloc[:, j] = col / max_val if max_val != 0 else col * 0 + 1.0
        else:
            min_val = col.min()
            safe_col = col.replace(0, 1e-10)
            norm.iloc[:, j] = min_val / safe_col if min_val != 0 else 1.0 / (safe_col / safe_col.min())
    return norm.clip(lower=0.0)


def _make_result(scores: pd.Series, ascending: bool) -> dict:
    ranking = list(scores.sort_values(ascending=ascending).index)
    return {
        "ranking": ranking,
        "scores": scores.to_dict(),
        "ranks": {alt: i + 1 for i, alt in enumerate(ranking)},
    }


# ---------------------------------------------------------------------------
# MCDM methods (each returns {"ranking", "scores", "ranks"})
# ---------------------------------------------------------------------------

def wsm(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """WSM (Weighted Sum Model / SAW): weighted sum of linearly normalised values."""
    norm = _linear_normalize(matrix, criteria_types)
    scores = (norm * weights).sum(axis=1)
    return _make_result(scores, ascending=False)


def wpm(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """WPM (Weighted Product Model): weighted product of normalised values."""
    norm = _linear_normalize(matrix, criteria_types).clip(lower=1e-10)
    scores_arr = np.prod(np.power(norm.values, weights), axis=1)
    scores = pd.Series(scores_arr, index=matrix.index)
    return _make_result(scores, ascending=False)


def topsis(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """TOPSIS: vector normalisation → weighted matrix → ideal distances → closeness coefficient."""
    m = matrix.values.astype(float)
    n_crits = m.shape[1]

    col_norms = np.sqrt((m ** 2).sum(axis=0))
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    weighted_m = (m / col_norms) * weights

    ideal_best = np.array([
        weighted_m[:, j].max() if criteria_types[j] == 'benefit' else weighted_m[:, j].min()
        for j in range(n_crits)
    ])
    ideal_worst = np.array([
        weighted_m[:, j].min() if criteria_types[j] == 'benefit' else weighted_m[:, j].max()
        for j in range(n_crits)
    ])

    dist_best = np.sqrt(((weighted_m - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_m - ideal_worst) ** 2).sum(axis=1))
    denom = dist_best + dist_worst
    denom = np.where(denom == 0, 1e-10, denom)
    closeness = dist_worst / denom

    scores = pd.Series(closeness, index=matrix.index)
    return _make_result(scores, ascending=False)


def vikor(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """VIKOR: compromise ranking via utility (S), regret (R), and index Q."""
    m = matrix.values.astype(float)
    n_crits = m.shape[1]

    f_star = np.array([
        m[:, j].max() if criteria_types[j] == 'benefit' else m[:, j].min()
        for j in range(n_crits)
    ])
    f_minus = np.array([
        m[:, j].min() if criteria_types[j] == 'benefit' else m[:, j].max()
        for j in range(n_crits)
    ])

    denom = np.where(f_star - f_minus == 0, 1e-10, f_star - f_minus)
    diff = (f_star - m) / denom

    S = (weights * diff).sum(axis=1)
    R = (weights * diff).max(axis=1)

    v = 0.5
    S_range = S.max() - S.min() or 1e-10
    R_range = R.max() - R.min() or 1e-10
    Q = v * (S - S.min()) / S_range + (1 - v) * (R - R.min()) / R_range

    scores = pd.Series(Q, index=matrix.index)
    return _make_result(scores, ascending=True)


def electre_i(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """ELECTRE I: pairwise concordance/discordance → net concordance flow ranking."""
    m = matrix.values.astype(float)
    n_alts, n_crits = m.shape

    norm_m = np.zeros_like(m)
    for j in range(n_crits):
        col_range = m[:, j].max() - m[:, j].min()
        if col_range == 0:
            norm_m[:, j] = 0.5
        elif criteria_types[j] == 'benefit':
            norm_m[:, j] = (m[:, j] - m[:, j].min()) / col_range
        else:
            norm_m[:, j] = (m[:, j].max() - m[:, j]) / col_range

    concordance = np.zeros((n_alts, n_alts))
    for i in range(n_alts):
        for k in range(n_alts):
            if i != k:
                concordance[i, k] = weights[norm_m[i] >= norm_m[k]].sum()

    net_flow = concordance.sum(axis=1) - concordance.sum(axis=0)
    scores = pd.Series(net_flow, index=matrix.index)
    return _make_result(scores, ascending=False)


def ahp_ranking(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """AHP as ranking method: weighted sum of normalised scores (same as WSM)."""
    return wsm(matrix, weights, criteria_types)


# ---------------------------------------------------------------------------
# AHP weight derivation (pairwise comparison matrix → priority vector)
# ---------------------------------------------------------------------------

def ahp_compute_weights(comparison_matrix) -> tuple:
    """Derive AHP priority weights and Consistency Ratio from pairwise comparison matrix.

    Returns:
        (weights: np.ndarray, CR: float)
    """
    cm = np.array(comparison_matrix, dtype=float)
    n = len(cm)
    geo_means = np.prod(cm, axis=1) ** (1.0 / n)
    weights = geo_means / geo_means.sum()

    lambda_max = float((cm @ weights / weights).mean())
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_VALUES.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return weights, CR


# ---------------------------------------------------------------------------
# Entropy-based weights (computed from decision matrix)
# ---------------------------------------------------------------------------

def entropy_weights(matrix: pd.DataFrame, criteria_types: np.ndarray) -> np.ndarray:
    """Compute objective weights from decision matrix using Shannon entropy."""
    m = matrix.values.astype(float)
    m_pos = np.abs(m) + 1e-10
    col_sums = m_pos.sum(axis=0)
    p = m_pos / col_sums
    # For cost criteria invert normalisation
    for j, ct in enumerate(criteria_types):
        if ct == 'cost':
            p[:, j] = (1.0 / (m_pos[:, j])) / (1.0 / m_pos[:, j]).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.where(p > 0, p * np.log(p), 0).sum(axis=0) / np.log(len(m))
    diversity = 1.0 - entropy
    total = diversity.sum()
    return diversity / total if total > 0 else np.ones(m.shape[1]) / m.shape[1]


# ---------------------------------------------------------------------------
# Core runner (DB-free inner loop)
# ---------------------------------------------------------------------------

_METHODS = {
    "WSM": wsm,
    "WPM": wpm,
    "TOPSIS": topsis,
    "VIKOR": vikor,
    "ELECTRE I": electre_i,
    "AHP": ahp_ranking,
}


def run_methods_only(matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> dict:
    """Run all 6 MCDM methods on provided data — no DB access."""
    ranking_table, scores_table, ranks_table = {}, {}, {}
    for name, func in _METHODS.items():
        try:
            res = func(matrix, weights, criteria_types)
            ranking_table[name] = " → ".join(res["ranking"])
            scores_table[name] = res["scores"]
            ranks_table[name] = res["ranks"]
        except Exception as exc:
            ranking_table[name] = f"Помилка: {exc}"
            scores_table[name] = {}
            ranks_table[name] = {}
    return {"success": True, "ranking_table": ranking_table,
            "scores_table": scores_table, "ranks_table": ranks_table}


# ---------------------------------------------------------------------------
# Main entry point (loads from DB, optionally accepts overrides)
# ---------------------------------------------------------------------------

def run_all_mcdm(engine, task_id: int, override_matrix=None, override_weights=None) -> dict:
    """Run all 6 MCDM methods for *task_id*.

    Args:
        engine: SQLAlchemy engine
        task_id: task primary key
        override_matrix: optional DataFrame (alternatives × criteria) — used for what-if
        override_weights: optional weight array aligned to criteria order

    Returns:
        dict with keys: success, ranking_table, scores_table, ranks_table,
                        matrix, weights, criteria, alternatives, criteria_types
    """
    with engine.connect() as conn:
        criteria = pd.read_sql(
            text("SELECT id, name, type FROM criteria WHERE task_id = :tid ORDER BY id"),
            conn, params={"tid": task_id},
        )
        alternatives = pd.read_sql(
            text("SELECT id, name FROM alternatives WHERE task_id = :tid ORDER BY id"),
            conn, params={"tid": task_id},
        )

    if criteria.empty or alternatives.empty:
        return {"success": False, "message": "Відсутні критерії або альтернативи для цієї задачі."}

    criteria_types = criteria["type"].values

    # ---- Build decision matrix ----
    if override_matrix is not None:
        matrix = override_matrix.copy()
        matrix = matrix.reindex(columns=criteria["name"].tolist()).astype(float)
    else:
        with engine.connect() as conn:
            evaluations = pd.read_sql(
                text("""
                    SELECT e.alternative_id, e.criterion_id, e.value
                    FROM evaluations e
                    JOIN alternatives a ON e.alternative_id = a.id
                    WHERE a.task_id = :tid
                """),
                conn, params={"tid": task_id},
            )
        if evaluations.empty:
            return {"success": False, "message": "Матриця оцінок порожня. Заповніть матрицю оцінок."}

        crit_map = dict(zip(criteria["id"], criteria["name"]))
        alt_map = dict(zip(alternatives["id"], alternatives["name"]))
        matrix = pd.DataFrame(index=alternatives["name"].tolist(),
                               columns=criteria["name"].tolist(), dtype=float)
        for _, row in evaluations.iterrows():
            a = alt_map.get(row["alternative_id"])
            c = crit_map.get(row["criterion_id"])
            if a and c:
                matrix.loc[a, c] = float(row["value"])

        if matrix.isnull().any().any():
            missing = int(matrix.isnull().sum().sum())
            return {"success": False,
                    "message": f"Матриця заповнена не повністю: {missing} пропущених значень."}

    # ---- Load weights ----
    if override_weights is not None:
        weights = np.array(override_weights, dtype=float)
    else:
        with engine.connect() as conn:
            w_df = pd.read_sql(
                text("SELECT criterion_id, weight_value FROM weights WHERE task_id = :tid"),
                conn, params={"tid": task_id},
            )
        weights = np.ones(len(criteria)) / len(criteria)
        if not w_df.empty:
            idx_map = {int(cid): i for i, cid in enumerate(criteria["id"])}
            w_arr = np.ones(len(criteria)) / len(criteria)
            found = False
            for _, wrow in w_df.iterrows():
                idx = idx_map.get(int(wrow["criterion_id"]))
                if idx is not None:
                    w_arr[idx] = float(wrow["weight_value"])
                    found = True
            if found:
                total = w_arr.sum()
                weights = w_arr / total if total > 0 else w_arr

    if len(weights) != len(criteria):
        weights = np.ones(len(criteria)) / len(criteria)
    s = weights.sum()
    if s > 0:
        weights = weights / s

    result = run_methods_only(matrix, weights, criteria_types)
    result.update({
        "matrix": matrix.to_dict(),
        "weights": weights.tolist(),
        "criteria": criteria["name"].tolist(),
        "alternatives": alternatives["name"].tolist(),
        "criteria_types": criteria_types.tolist(),
    })
    return result
