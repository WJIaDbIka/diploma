# MCDM Decision Support System

Web-based multi-criteria decision making (MCDM) system built with Python and Streamlit.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run main.py
```

Then open http://localhost:8501 in your browser.

## Workflow

1. **Створити задачу** — create a decision task with a name and description.
2. **Редагувати задачу** — add criteria (benefit/cost), add alternatives, fill the evaluation matrix.
3. **Ваги критеріїв** — set criterion weights manually, via AHP pairwise comparison, or via entropy method.
4. **MCDM-розрахунки** — run all six methods at once; results are saved to the database.
5. **Порівняння методів** — view Borda consensus, Spearman/Kendall correlations, heatmap, and generate a PDF report.
6. **What-if аналіз** — modify matrix values in-browser and compare the resulting rankings to the base scenario; run sensitivity analysis on each criterion weight.
7. **Історія рішень** — browse all saved computation results with task-level filtering.

## Implemented MCDM Methods

### WSM — Weighted Sum Model (SAW)

Normalize each criterion linearly: benefit → `x/max`, cost → `min/x`.
Compute score: `S_i = Σ w_j · r_ij`. Rank descending.

### WPM — Weighted Product Model

Same linear normalization as WSM.
Score: `P_i = Π r_ij^w_j`. Rank descending.

### TOPSIS — Technique for Order Preference by Similarity to Ideal Solution

1. Vector-normalize: `r_ij = x_ij / √(Σ x_ij²)`
2. Weighted matrix: `v_ij = w_j · r_ij`
3. Ideal best `A*` and ideal worst `A⁻` per criterion type.
4. Euclidean distances `D*_i`, `D⁻_i`.
5. Closeness coefficient: `C_i = D⁻_i / (D*_i + D⁻_i)`. Rank descending.

### VIKOR — VIseKriterijumska Optimizacija I Kompromisno Resenje

1. Ideal `f*_j` and anti-ideal `f⁻_j` per criterion.
2. Utility: `S_i = Σ w_j·(f*_j − f_ij)/(f*_j − f⁻_j)`
3. Regret: `R_i = max_j [w_j·(f*_j − f_ij)/(f*_j − f⁻_j)]`
4. Compromise index (v = 0.5): `Q_i = v·(S_i − S*)/(S⁻ − S*) + (1−v)·(R_i − R*)/(R⁻ − R*)`
5. Rank ascending by `Q`.

### ELECTRE I — Elimination and Choice Translating Reality

1. Min-max normalize the decision matrix; apply benefit/cost direction.
2. For every pair (a, b): concordance `c(a,b) = Σ w_j` over criteria where a ≥ b.
3. Net concordance flow: `φ(a) = Σ_b c(a,b) − Σ_b c(b,a)`. Rank descending.

### AHP — Analytic Hierarchy Process (as ranking)

When used as a ranking method, AHP aggregates alternative scores via a weighted sum of normalized criteria values (equivalent to WSM).

**Weight derivation via pairwise comparison (⚖️ Ваги критеріїв tab):**
1. Build n×n Saaty comparison matrix `C`.
2. Priority vector: geometric-mean method — `w_i = (Π C_ij)^(1/n) / Σ ...`
3. Consistency: `λ_max = mean(Cw/w)`, `CI = (λ_max − n)/(n−1)`, `CR = CI/RI`.
4. Acceptable if `CR ≤ 0.1`.

## Database Schema

SQLite file `mcdm_decisions.db` with six tables:

| Table | Description |
|---|---|
| `tasks` | Decision tasks |
| `criteria` | Criteria per task (benefit/cost type) |
| `alternatives` | Alternatives per task |
| `evaluations` | Decision matrix values |
| `weights` | Criterion weights per task |
| `mcdm_results` | Saved ranking results |

## Tech Stack

- **Frontend**: Streamlit 1.56
- **Data**: Pandas, NumPy, SciPy
- **Charts**: Plotly
- **Database**: SQLite via SQLAlchemy 2.x
- **PDF**: fpdf2 with DejaVuSans (Cyrillic support)
