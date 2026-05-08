import pandas as pd
import numpy as np
from sqlalchemy import text

def normalize_matrix(matrix, criteria_types):
    norm_matrix = matrix.copy().astype(float)
    for j, crit_type in enumerate(criteria_types):
        col = matrix.iloc[:, j]
        if crit_type == 'benefit':
            norm_matrix.iloc[:, j] = col / col.max()
        else:
            norm_matrix.iloc[:, j] = col.min() / col
    return norm_matrix

def saw(matrix, weights, criteria_types):
    norm = normalize_matrix(matrix, criteria_types)
    scores = (norm * weights).sum(axis=1)
    return scores.sort_values(ascending=False)

def wpm(matrix, weights, criteria_types):
    norm = normalize_matrix(matrix, criteria_types)
    scores = np.prod(np.power(norm, weights), axis=1)
    return scores.sort_values(ascending=False)

def topsis(matrix, weights, criteria_types):
    norm = normalize_matrix(matrix, criteria_types)
    weighted = norm * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    closeness = dist_worst / (dist_best + dist_worst)
    return closeness.sort_values(ascending=False)

def vikor(matrix, weights, criteria_types):
    norm = normalize_matrix(matrix, criteria_types)
    f_max = norm.max(axis=0)
    f_min = norm.min(axis=0)
    S = (weights * (f_max - norm) / (f_max - f_min)).sum(axis=1)
    R = (weights * (f_max - norm) / (f_max - f_min)).max(axis=1)
    Q = 0.5 * (S - S.min()) / (S.max() - S.min()) + 0.5 * (R - R.min()) / (R.max() - R.min())
    return Q.sort_values(ascending=True)

def promethee_ii(matrix, weights, criteria_types):
    m = matrix.shape[0]
    phi = np.zeros(m)
    for i in range(m):
        for j in range(m):
            if i != j:
                diff = matrix.iloc[i] - matrix.iloc[j]
                pref = np.where(criteria_types == 'benefit', diff, -diff)
                pref = np.maximum(0, pref)
                phi[i] += np.sum(weights * pref)
    phi = phi / (m - 1)
    return pd.Series(phi, index=matrix.index).sort_values(ascending=False)

def ahp(matrix, weights, criteria_types):
    norm = normalize_matrix(matrix, criteria_types)
    scores = (norm * weights).sum(axis=1)
    return scores.sort_values(ascending=False)

def run_all_mcdm(engine, task_id):
    """Головний метод: виконує всі 6 MCDM-методів"""
    # Завантаження даних
    criteria = pd.read_sql(f"SELECT * FROM criteria WHERE task_id = {task_id}", engine)
    alternatives = pd.read_sql(f"SELECT * FROM alternatives WHERE task_id = {task_id}", engine)
    evaluations = pd.read_sql(f"""
        SELECT e.* FROM evaluations e
        JOIN alternatives a ON e.alternative_id = a.id
        WHERE a.task_id = {task_id}
    """, engine)

    if criteria.empty or alternatives.empty or evaluations.empty:
        return {"success": False, "message": "Недостатньо даних. Перевірте, що є критерії, альтернативи та повністю заповнена матриця оцінок."}

    # Формування матриці рішень
    matrix = evaluations.pivot(index='alternative_id', columns='criterion_id', values='value')
    matrix = matrix.reindex(alternatives['id']).reset_index(drop=True)
    matrix.index = alternatives['name'].values
    criteria_types = criteria.set_index('id')['type'].loc[matrix.columns].values

    # Ваги (якщо немає — рівномірні)
    weights_df = pd.read_sql(f"SELECT * FROM weights WHERE task_id = {task_id} ORDER BY id DESC LIMIT 1", engine)
    if weights_df.empty:
        weights = np.ones(len(criteria)) / len(criteria)
    else:
        weights = weights_df['weight_value'].values.astype(float)

    methods = {
        "SAW": saw,
        "WPM": wpm,
        "TOPSIS": topsis,
        "VIKOR": vikor,
        "PROMETHEE II": promethee_ii,
        "AHP": ahp
    }

    ranking_table = {}
    for name, func in methods.items():
        try:
            result = func(matrix, weights, criteria_types)
            ranking_table[name] = " → ".join(result.index)
        except Exception as e:
            ranking_table[name] = f"Помилка: {str(e)}"

    return {
        "success": True,
        "ranking_table": ranking_table,
        "matrix": matrix.to_dict()
    }