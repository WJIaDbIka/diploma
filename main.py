"""MCDM Decision Support System — Streamlit web application."""

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from scipy.stats import kendalltau, spearmanr
from sqlalchemy import create_engine, text

from mcdm_methods.mcdm_calculator import (
    ahp_compute_weights,
    entropy_weights,
    run_all_mcdm,
    run_methods_only,
)

# ============================================================
# App config
# ============================================================
st.set_page_config(
    page_title="MCDM Decision Support System",
    page_icon="📊",
    layout="wide",
)
st.title("🧠 Веб-система підтримки прийняття управлінських рішень")
st.subheader("на основі методів багатокритеріального аналізу (MCDM)")


# ============================================================
# Database
# ============================================================
@st.cache_resource
def get_db_engine():
    return create_engine("sqlite:///mcdm_decisions.db", echo=False)


engine = get_db_engine()

_DDL = [
    """CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS criteria (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        name TEXT NOT NULL,
        type TEXT CHECK(type IN ('benefit','cost')),
        unit TEXT,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )""",
    """CREATE TABLE IF NOT EXISTS alternatives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        name TEXT NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )""",
    """CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alternative_id INTEGER,
        criterion_id INTEGER,
        value REAL NOT NULL,
        FOREIGN KEY(alternative_id) REFERENCES alternatives(id),
        FOREIGN KEY(criterion_id) REFERENCES criteria(id)
    )""",
    """CREATE TABLE IF NOT EXISTS weights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        criterion_id INTEGER,
        weight_value REAL NOT NULL,
        is_automatic INTEGER DEFAULT 0,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )""",
    """CREATE TABLE IF NOT EXISTS mcdm_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        method_name TEXT NOT NULL,
        ranking TEXT,
        computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    )""",
]
with engine.connect() as _conn:
    for _sql in _DDL:
        _conn.execute(text(_sql))
    _conn.commit()


# ============================================================
# Helpers
# ============================================================

def load_tasks() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT id, name FROM tasks ORDER BY created_at DESC"), conn
        )


def task_selectbox(label: str = "Оберіть задачу", key: str = "task_sel") -> int | None:
    df = load_tasks()
    if df.empty:
        st.warning("Спочатку створіть задачу (розділ «Створити задачу»).")
        return None
    return st.selectbox(
        label,
        options=df["id"].tolist(),
        format_func=lambda x: df[df["id"] == x]["name"].iloc[0],
        key=key,
    )


def load_criteria(task_id: int) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT id, name, type, unit FROM criteria WHERE task_id = :tid ORDER BY id"),
            conn, params={"tid": task_id},
        )


def load_alternatives(task_id: int) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("SELECT id, name FROM alternatives WHERE task_id = :tid ORDER BY id"),
            conn, params={"tid": task_id},
        )


def load_evaluations(task_id: int) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT e.alternative_id, e.criterion_id, e.value,
                       a.name AS alt_name, c.name AS crit_name
                FROM evaluations e
                JOIN alternatives a ON e.alternative_id = a.id
                JOIN criteria c ON e.criterion_id = c.id
                WHERE a.task_id = :tid
            """),
            conn, params={"tid": task_id},
        )


def build_matrix(criteria: pd.DataFrame, alternatives: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=alternatives["name"].tolist(),
                          columns=criteria["name"].tolist(), dtype=float)
    for _, row in evaluations.iterrows():
        matrix.loc[row["alt_name"], row["crit_name"]] = float(row["value"])
    return matrix


def load_weights(task_id: int, criteria: pd.DataFrame) -> np.ndarray:
    with engine.connect() as conn:
        w_df = pd.read_sql(
            text("SELECT criterion_id, weight_value FROM weights WHERE task_id = :tid"),
            conn, params={"tid": task_id},
        )
    n = len(criteria)
    weights = np.ones(n) / n
    if not w_df.empty:
        idx_map = {int(cid): i for i, cid in enumerate(criteria["id"])}
        w_arr = np.ones(n) / n
        found = False
        for _, r in w_df.iterrows():
            idx = idx_map.get(int(r["criterion_id"]))
            if idx is not None:
                w_arr[idx] = float(r["weight_value"])
                found = True
        if found:
            s = w_arr.sum()
            weights = w_arr / s if s > 0 else w_arr
    return weights


def save_weights(task_id: int, criteria: pd.DataFrame, weights: np.ndarray, is_automatic: int = 0):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM weights WHERE task_id = :tid"), {"tid": task_id})
        for i, (_, row) in enumerate(criteria.iterrows()):
            conn.execute(
                text("INSERT INTO weights (task_id, criterion_id, weight_value, is_automatic) "
                     "VALUES (:tid, :cid, :w, :auto)"),
                {"tid": task_id, "cid": int(row["id"]), "w": float(weights[i]), "auto": is_automatic},
            )
        conn.commit()


def _rank_symbol(base: int, new: int) -> str:
    if new < base:
        return f"↑ ({base}→{new})"
    if new > base:
        return f"↓ ({base}→{new})"
    return f"= ({base})"


# ============================================================
# PDF report generation
# ============================================================
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def generate_pdf_report(results: dict, task_name: str) -> bytes:
    pdf = FPDF()
    pdf.add_font("DV", "", FONT_PATH)
    pdf.add_font("DV", "B", FONT_BOLD)

    def h1(text_: str):
        pdf.set_font("DV", "B", 14)
        pdf.ln(4)
        pdf.multi_cell(0, 8, text_)
        pdf.ln(2)

    def h2(text_: str):
        pdf.set_font("DV", "B", 11)
        pdf.ln(3)
        pdf.multi_cell(0, 7, text_)

    def body(text_: str):
        pdf.set_font("DV", "", 10)
        pdf.multi_cell(0, 6, text_)

    def table(df: pd.DataFrame, col_w: int = 35):
        pdf.set_font("DV", "B", 9)
        all_cols = [""] + list(df.columns)
        for col in all_cols:
            pdf.cell(col_w, 7, str(col)[:20], border=1)
        pdf.ln()
        pdf.set_font("DV", "", 9)
        for idx, row in df.iterrows():
            pdf.cell(col_w, 6, str(idx)[:20], border=1)
            for val in row:
                pdf.cell(col_w, 6, str(val)[:20], border=1)
            pdf.ln()
        pdf.ln(2)

    NX, NY = XPos.LMARGIN, YPos.NEXT
    alts = results.get("alternatives", [])
    crits = results.get("criteria", [])

    # --- Title page ---
    pdf.add_page()
    pdf.set_font("DV", "B", 18)
    pdf.cell(0, 12, "MCDM — Звіт про прийняте рішення", new_x=NX, new_y=NY, align="C")
    pdf.set_font("DV", "", 12)
    pdf.cell(0, 8, f"Задача: {task_name}", new_x=NX, new_y=NY)
    pdf.cell(0, 8, f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=NX, new_y=NY)
    pdf.cell(0, 8, f"Альтернатив: {len(alts)},  критеріїв: {len(crits)}", new_x=NX, new_y=NY)
    pdf.ln(6)

    # --- 1: Decision matrix ---
    h1("1. Матриця рішень")
    mat_dict = results.get("matrix", {})
    if mat_dict and alts and crits:
        mat_df = pd.DataFrame(mat_dict)
        mat_df.index.name = None
        try:
            table(mat_df)
        except Exception:
            body(str(mat_dict))

    # --- 2: Weights ---
    h1("2. Ваги критеріїв")
    weights = results.get("weights", [])
    if weights and crits:
        w_df = pd.DataFrame({"Критерій": crits, "Вага": [round(w, 4) for w in weights]})
        for _, r in w_df.iterrows():
            body(f"  {r['Критерій']}: {r['Вага']:.4f}")
    pdf.ln(2)

    # --- 3: Method results ---
    h1("3. Результати кожного методу")
    ranks_table = results.get("ranks_table", {})
    scores_table = results.get("scores_table", {})
    for method, ranking_str in results.get("ranking_table", {}).items():
        h2(f"Метод: {method}")
        body(f"Ранжування: {ranking_str}")
        if alts and method in scores_table:
            sc = scores_table[method]
            for alt in alts:
                v = sc.get(alt, "-")
                body(f"  {alt}: {round(v, 5) if isinstance(v, float) else v}")
        pdf.ln(1)

    # --- 4: Borda consensus ---
    pdf.add_page()
    h1("4. Консенсусне ранжування (метод Борда)")
    if alts and ranks_table:
        borda = {a: sum(ranks_table[m].get(a, len(alts)) for m in ranks_table) for a in alts}
        borda_sorted = sorted(borda.items(), key=lambda x: x[1])
        for rank, (alt, score) in enumerate(borda_sorted, 1):
            body(f"  {rank}. {alt} — бал Борда: {score}")
    pdf.ln(2)

    # --- 5: Correlation summary ---
    h1("5. Узгодженість між методами")
    if len(ranks_table) >= 2:
        methods_list = [m for m in ranks_table if ranks_table[m]]
        vecs = {m: [ranks_table[m].get(a, len(alts)) for a in alts] for m in methods_list}
        body("Кореляція Спірмена між методами:")
        for i, m1 in enumerate(methods_list):
            for m2 in methods_list[i + 1:]:
                rho, _ = spearmanr(vecs[m1], vecs[m2])
                body(f"  {m1} vs {m2}: ρ = {rho:.3f}")
    pdf.ln(2)

    # --- 6: Recommendation ---
    h1("6. Рекомендоване рішення")
    if alts and ranks_table:
        borda = {a: sum(ranks_table[m].get(a, len(alts)) for m in ranks_table) for a in alts}
        best_alt = min(borda, key=borda.get)
        top1_count = sum(1 for m in ranks_table if ranks_table[m].get(best_alt) == 1)
        body(f"Рекомендується: {best_alt}")
        body(f"Бал Борда: {borda[best_alt]} (найменший = найкращий)")
        body(f"Посідає 1-е місце у {top1_count} з {len(ranks_table)} методів.")

    return bytes(pdf.output())


# ============================================================
# Sidebar navigation
# ============================================================
with st.sidebar:
    st.header("Навігація")
    page = st.radio(
        "Оберіть розділ",
        [
            "🏠 Головна",
            "📋 Створити задачу",
            "✏️ Редагувати задачу",
            "⚖️ Ваги критеріїв",
            "📊 MCDM-розрахунки",
            "📈 Порівняння методів",
            "🔄 What-if аналіз",
            "📜 Історія рішень",
        ],
    )

# ============================================================
# 🏠 Головна
# ============================================================
if page == "🏠 Головна":
    st.header("🏠 Головна сторінка")
    try:
        with engine.connect() as conn:
            n_tasks = conn.execute(text("SELECT COUNT(*) FROM tasks")).scalar()
            n_crit = conn.execute(text("SELECT COUNT(*) FROM criteria")).scalar()
            n_alt = conn.execute(text("SELECT COUNT(*) FROM alternatives")).scalar()
            n_results = conn.execute(text("SELECT COUNT(*) FROM mcdm_results")).scalar()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Методів MCDM", "6")
        col2.metric("Задач у БД", n_tasks)
        col3.metric("Критеріїв", n_crit)
        col4.metric("Альтернатив", n_alt)
        col5.metric("Розрахунків", n_results)
    except Exception as e:
        st.error(f"Помилка завантаження статистики: {e}")

    st.markdown("""
    ### Як користуватися системою
    1. **Створити задачу** — введіть назву та опис задачі прийняття рішень.
    2. **Редагувати задачу** — додайте критерії, альтернативи і заповніть матрицю оцінок.
    3. **Ваги критеріїв** — задайте ваги вручну, методом AHP або ентропії.
    4. **MCDM-розрахунки** — запустіть усі 6 методів (WSM, WPM, TOPSIS, VIKOR, ELECTRE I, AHP).
    5. **Порівняння методів** — перегляньте Borda-консенсус, кореляції, теплову карту та згенеруйте PDF.
    6. **What-if аналіз** — змоделюйте зміни в матриці та перегляньте нові результати.
    7. **Історія рішень** — перегляньте всі збережені розрахунки.
    """)

# ============================================================
# 📋 Створити задачу
# ============================================================
elif page == "📋 Створити задачу":
    st.header("📋 Створення нової задачі")
    with st.form("create_task_form"):
        task_name = st.text_input("Назва задачі*", placeholder="Вибір постачальника промислового обладнання")
        task_desc = st.text_area("Опис задачі", height=100)
        submitted = st.form_submit_button("✅ Створити задачу")
    if submitted:
        if not task_name.strip():
            st.error("Введіть назву задачі.")
        else:
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text("INSERT INTO tasks (name, description) VALUES (:name, :desc)"),
                        {"name": task_name.strip(), "desc": task_desc},
                    )
                    conn.commit()
                st.success(f"Задача «{task_name}» успішно створена!")
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")

# ============================================================
# ✏️ Редагувати задачу
# ============================================================
elif page == "✏️ Редагувати задачу":
    st.header("✏️ Редагування задачі")
    tid = task_selectbox(key="edit_task_sel")
    if tid is None:
        st.stop()

    tasks_df = load_tasks()
    task_name_current = tasks_df[tasks_df["id"] == tid]["name"].iloc[0]
    st.subheader(f"Задача: {task_name_current}")

    # Delete task
    with st.expander("🗑️ Видалити задачу", expanded=False):
        st.warning("Це незворотна операція! Усі дані задачі будуть видалені.")
        if st.button("Підтвердити видалення", type="primary"):
            try:
                with engine.connect() as conn:
                    conn.execute(text("""
                        DELETE FROM evaluations WHERE alternative_id IN
                        (SELECT id FROM alternatives WHERE task_id = :tid)
                    """), {"tid": tid})
                    conn.execute(text("DELETE FROM weights WHERE task_id = :tid"), {"tid": tid})
                    conn.execute(text("DELETE FROM mcdm_results WHERE task_id = :tid"), {"tid": tid})
                    conn.execute(text("DELETE FROM criteria WHERE task_id = :tid"), {"tid": tid})
                    conn.execute(text("DELETE FROM alternatives WHERE task_id = :tid"), {"tid": tid})
                    conn.execute(text("DELETE FROM tasks WHERE id = :tid"), {"tid": tid})
                    conn.commit()
                st.success("Задачу видалено.")
                st.rerun()
            except Exception as e:
                st.error(f"Помилка: {e}")

    tab1, tab2, tab3 = st.tabs(["Критерії", "Альтернативи", "Матриця оцінок"])

    with tab1:
        with st.form("add_criterion"):
            c1, c2, c3 = st.columns(3)
            crit_name = c1.text_input("Назва критерію*")
            crit_type = c2.selectbox("Тип", ["benefit", "cost"])
            crit_unit = c3.text_input("Одиниця вимірювання")
            if st.form_submit_button("Додати критерій"):
                if not crit_name.strip():
                    st.error("Введіть назву критерію.")
                else:
                    try:
                        with engine.connect() as conn:
                            conn.execute(
                                text("INSERT INTO criteria (task_id, name, type, unit) "
                                     "VALUES (:t, :n, :ty, :u)"),
                                {"t": tid, "n": crit_name.strip(), "ty": crit_type, "u": crit_unit},
                            )
                            conn.commit()
                        st.success("Критерій додано!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Помилка: {e}")
        crit_df = load_criteria(tid)
        if not crit_df.empty:
            st.dataframe(crit_df.drop(columns=["id"]), use_container_width=True)
        else:
            st.info("Критеріїв ще немає.")

    with tab2:
        with st.form("add_alternative"):
            alt_name = st.text_input("Назва альтернативи*")
            if st.form_submit_button("Додати альтернативу"):
                if not alt_name.strip():
                    st.error("Введіть назву альтернативи.")
                else:
                    try:
                        with engine.connect() as conn:
                            conn.execute(
                                text("INSERT INTO alternatives (task_id, name) VALUES (:t, :n)"),
                                {"t": tid, "n": alt_name.strip()},
                            )
                            conn.commit()
                        st.success("Альтернативу додано!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Помилка: {e}")
        alt_df = load_alternatives(tid)
        if not alt_df.empty:
            st.dataframe(alt_df.drop(columns=["id"]), use_container_width=True)
        else:
            st.info("Альтернатив ще немає.")

    with tab3:
        criteria = load_criteria(tid)
        alternatives = load_alternatives(tid)
        if criteria.empty or alternatives.empty:
            st.warning("Спочатку додайте критерії та альтернативи.")
        else:
            evaluations = load_evaluations(tid)
            matrix = build_matrix(criteria, alternatives, evaluations)
            edited = st.data_editor(
                matrix,
                use_container_width=True,
                num_rows="fixed",
                key="matrix_editor",
                column_config={c: st.column_config.NumberColumn(c) for c in criteria["name"].tolist()},
            )
            if st.button("💾 Зберегти матрицю оцінок"):
                try:
                    with engine.connect() as conn:
                        conn.execute(
                            text("DELETE FROM evaluations WHERE alternative_id IN "
                                 "(SELECT id FROM alternatives WHERE task_id = :tid)"),
                            {"tid": tid},
                        )
                        conn.commit()
                    with engine.connect() as conn:
                        for alt_name_row, row in edited.iterrows():
                            alt_id = int(alternatives[alternatives["name"] == alt_name_row]["id"].iloc[0])
                            for crit_name_col, value in row.items():
                                if pd.notna(value):
                                    crit_id = int(criteria[criteria["name"] == crit_name_col]["id"].iloc[0])
                                    conn.execute(
                                        text("INSERT INTO evaluations (alternative_id, criterion_id, value) "
                                             "VALUES (:a, :c, :v)"),
                                        {"a": alt_id, "c": crit_id, "v": float(value)},
                                    )
                        conn.commit()
                    st.success("✅ Матрицю оцінок збережено!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Помилка збереження: {e}")

# ============================================================
# ⚖️ Ваги критеріїв
# ============================================================
elif page == "⚖️ Ваги критеріїв":
    st.header("⚖️ Ваги критеріїв")
    tid = task_selectbox(key="weights_task_sel")
    if tid is None:
        st.stop()

    criteria = load_criteria(tid)
    if criteria.empty:
        st.warning("Спочатку додайте критерії до задачі.")
        st.stop()

    n_c = len(criteria)
    crit_names = criteria["name"].tolist()

    tab_manual, tab_ahp, tab_entropy = st.tabs(
        ["Ручне введення", "Метод AHP (Saaty)", "Метод ентропії"]
    )

    # ---- Tab 1: Manual ----
    with tab_manual:
        st.write("Введіть ваги для кожного критерію. Сума повинна дорівнювати 1.0.")
        current_weights = load_weights(tid, criteria)
        manual_w = []
        cols = st.columns(min(n_c, 4))
        for i, name in enumerate(crit_names):
            with cols[i % len(cols)]:
                w = st.number_input(
                    name, min_value=0.0, max_value=1.0,
                    value=float(round(current_weights[i], 4)),
                    step=0.01, format="%.4f", key=f"manual_w_{i}",
                )
                manual_w.append(w)
        total = sum(manual_w)
        st.metric("Сума ваг", f"{total:.4f}", delta=f"{total - 1.0:+.4f}")
        if abs(total - 1.0) > 0.001:
            st.warning("⚠️ Сума ваг не дорівнює 1.0. Натисніть «Нормалізувати» або відкоригуйте вручну.")
        c1, c2 = st.columns(2)
        if c1.button("🔄 Нормалізувати та зберегти"):
            norm = np.array(manual_w, dtype=float)
            s = norm.sum()
            if s == 0:
                st.error("Усі ваги нульові — неможливо нормалізувати.")
            else:
                norm /= s
                save_weights(tid, criteria, norm)
                st.success("Ваги нормалізовано та збережено!")
                st.rerun()
        if c2.button("💾 Зберегти (без нормалізації)"):
            save_weights(tid, criteria, np.array(manual_w))
            st.success("Ваги збережено!")
            st.rerun()

    # ---- Tab 2: AHP ----
    with tab_ahp:
        st.write("Заповніть матрицю попарних порівнянь (шкала Сааті 1/9…1…9).")
        st.info("Значення > 1: перший критерій важливіший. Значення < 1: другий важливіший.")

        SAATY_LABELS = ["1/9", "1/8", "1/7", "1/6", "1/5", "1/4", "1/3", "1/2",
                        "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        SAATY_VALUES = [1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2,
                        1, 2, 3, 4, 5, 6, 7, 8, 9]

        pairs = [(i, j) for i in range(n_c) for j in range(i + 1, n_c)]
        ahp_vals = {}
        cols_per_row = 2
        pair_cols = st.columns(cols_per_row)
        for idx, (i, j) in enumerate(pairs):
            with pair_cols[idx % cols_per_row]:
                sel = st.select_slider(
                    f"**{crit_names[i]}** vs **{crit_names[j]}**",
                    options=SAATY_LABELS,
                    value="1",
                    key=f"ahp_{tid}_{i}_{j}",
                )
                ahp_vals[(i, j)] = SAATY_VALUES[SAATY_LABELS.index(sel)]

        # Build full comparison matrix
        cm = np.ones((n_c, n_c))
        for (i, j), v in ahp_vals.items():
            cm[i, j] = v
            cm[j, i] = 1.0 / v

        ahp_weights, CR = ahp_compute_weights(cm)
        st.subheader("Результати AHP")
        w_df_ahp = pd.DataFrame({
            "Критерій": crit_names,
            "Вага AHP": [round(w, 4) for w in ahp_weights],
        })
        st.dataframe(w_df_ahp, use_container_width=True)
        if CR > 0.1:
            st.warning(f"⚠️ Коефіцієнт узгодженості CR = {CR:.3f} > 0.1 — матриця недостатньо узгоджена!")
        else:
            st.success(f"✅ CR = {CR:.3f} ≤ 0.1 — матриця узгоджена.")

        if st.button("💾 Зберегти ваги AHP"):
            save_weights(tid, criteria, ahp_weights)
            st.success("Ваги AHP збережено!")
            st.rerun()

    # ---- Tab 3: Entropy ----
    with tab_entropy:
        st.write("Об'єктивні ваги, розраховані автоматично з матриці оцінок.")
        evaluations = load_evaluations(tid)
        alternatives = load_alternatives(tid)
        if evaluations.empty or alternatives.empty:
            st.warning("Заповніть матрицю оцінок перед розрахунком ентропійних ваг.")
        else:
            matrix = build_matrix(criteria, alternatives, evaluations)
            if matrix.isnull().any().any():
                st.warning("Матриця оцінок неповна — заповніть усі значення.")
            else:
                crit_types = criteria["type"].values
                ent_w = entropy_weights(matrix, crit_types)
                w_df_ent = pd.DataFrame({
                    "Критерій": crit_names,
                    "Ентропійна вага": [round(w, 4) for w in ent_w],
                })
                st.dataframe(w_df_ent, use_container_width=True)
                if st.button("💾 Зберегти ентропійні ваги"):
                    save_weights(tid, criteria, ent_w, is_automatic=1)
                    st.success("Ентропійні ваги збережено!")
                    st.rerun()

# ============================================================
# 📊 MCDM-розрахунки
# ============================================================
elif page == "📊 MCDM-розрахунки":
    st.header("📊 Виконання MCDM-розрахунків")
    tid = task_selectbox(key="mcdm_task_sel")
    if tid is None:
        st.stop()

    criteria = load_criteria(tid)
    alternatives = load_alternatives(tid)
    evaluations = load_evaluations(tid)

    # Validation
    if criteria.empty:
        st.error("❌ Задача не має критеріїв. Додайте критерії.")
        st.stop()
    if alternatives.empty:
        st.error("❌ Задача не має альтернатив. Додайте альтернативи.")
        st.stop()
    if evaluations.empty:
        st.error("❌ Матриця оцінок порожня. Заповніть матрицю оцінок.")
        st.stop()

    matrix = build_matrix(criteria, alternatives, evaluations)
    missing_vals = int(matrix.isnull().sum().sum())
    if missing_vals > 0:
        st.error(f"❌ Матриця оцінок неповна: {missing_vals} пропущених значень.")
        st.stop()

    with engine.connect() as conn:
        has_weights = conn.execute(
            text("SELECT COUNT(*) FROM weights WHERE task_id = :tid"), {"tid": tid}
        ).scalar()
    if not has_weights:
        st.warning("⚠️ Ваги не задано — буде використано рівномірні ваги. Перейдіть у «Ваги критеріїв».")

    st.success(f"✅ Задача готова: {len(criteria)} критеріїв, {len(alternatives)} альтернатив.")

    if st.button("🚀 Виконати розрахунки за всіма 6 методами", type="primary"):
        with st.spinner("Виконується розрахунок 6 методів MCDM…"):
            try:
                results = run_all_mcdm(engine, tid)
                if not results["success"]:
                    st.error(results.get("message", "Невідома помилка."))
                    st.stop()

                st.success("✅ Розрахунки виконано успішно!")

                # Show ranking table
                df_res = pd.DataFrame(
                    list(results["ranking_table"].items()), columns=["Метод", "Ранжування"]
                )
                st.dataframe(df_res, use_container_width=True)

                # Save to DB (replace previous results for this task)
                with engine.connect() as conn:
                    conn.execute(
                        text("DELETE FROM mcdm_results WHERE task_id = :tid"), {"tid": tid}
                    )
                    for method, rank_str in results["ranking_table"].items():
                        conn.execute(
                            text("INSERT INTO mcdm_results (task_id, method_name, ranking, computed_at) "
                                 "VALUES (:tid, :m, :r, CURRENT_TIMESTAMP)"),
                            {"tid": tid, "m": method, "r": rank_str},
                        )
                    conn.commit()

                # Show scores
                st.subheader("Деталізовані оцінки альтернатив")
                alts = results["alternatives"]
                for method, scores in results["scores_table"].items():
                    if scores:
                        sc_df = pd.DataFrame(
                            [(a, round(scores.get(a, 0), 5)) for a in alts],
                            columns=["Альтернатива", "Оцінка"],
                        ).set_index("Альтернатива")
                        with st.expander(f"{method}"):
                            st.dataframe(sc_df, use_container_width=True)

                st.balloons()
            except Exception as e:
                st.error(f"Помилка розрахунку: {e}")

# ============================================================
# 📈 Порівняння методів
# ============================================================
elif page == "📈 Порівняння методів":
    st.header("📈 Порівняння результатів MCDM-методів")
    tid = task_selectbox(key="compare_task_sel")
    if tid is None:
        st.stop()

    criteria = load_criteria(tid)
    alternatives = load_alternatives(tid)
    evaluations = load_evaluations(tid)

    if criteria.empty or alternatives.empty or evaluations.empty:
        st.warning("Заповніть задачу та виконайте MCDM-розрахунки.")
        st.stop()

    matrix = build_matrix(criteria, alternatives, evaluations)
    if matrix.isnull().any().any():
        st.warning("Матриця оцінок неповна.")
        st.stop()

    with st.spinner("Розрахунок методів для порівняння…"):
        try:
            results = run_all_mcdm(engine, tid)
        except Exception as e:
            st.error(f"Помилка розрахунку: {e}")
            st.stop()

    if not results["success"]:
        st.error(results.get("message", ""))
        st.stop()

    alts = results["alternatives"]
    methods_list = [m for m in results["ranks_table"] if results["ranks_table"][m]]
    ranks_table = results["ranks_table"]
    scores_table = results["scores_table"]

    # --- 1: Ranking table ---
    st.subheader("1️⃣ Таблиця рангів альтернатив")
    rank_data = {m: [ranks_table[m].get(a, "-") for a in alts] for m in methods_list}
    rank_df = pd.DataFrame(rank_data, index=alts)
    rank_df.index.name = "Альтернатива"
    st.dataframe(rank_df.style.highlight_min(axis=0, color="#c8f7c5"), use_container_width=True)

    # --- 2: Bar chart — rank per alternative per method ---
    st.subheader("2️⃣ Порівняльна діаграма рангів")
    bar_rows = []
    for method in methods_list:
        for alt in alts:
            bar_rows.append({"Метод": method, "Альтернатива": alt,
                             "Ранг": ranks_table[method].get(alt, len(alts))})
    bar_df = pd.DataFrame(bar_rows)
    fig_bar = px.bar(
        bar_df, x="Альтернатива", y="Ранг", color="Метод", barmode="group",
        title="Позиція кожної альтернативи за кожним методом (менший ранг = краще)",
    )
    fig_bar.update_yaxes(autorange="reversed", dtick=1)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- 3: Radar chart — normalised scores ---
    st.subheader("3️⃣ Radar Chart нормалізованих оцінок (WSM)")
    if "WSM" in scores_table and scores_table["WSM"]:
        wsm_scores = scores_table["WSM"]
        max_score = max(wsm_scores.values()) or 1
        fig_radar = go.Figure()
        for alt in alts:
            val = wsm_scores.get(alt, 0)
            fig_radar.add_trace(go.Scatterpolar(
                r=[round(val / max_score, 3)] * len(methods_list),
                theta=methods_list,
                fill="toself",
                name=alt,
            ))
        fig_radar.update_layout(title="Відносна оцінка WSM (нормалізована) по методах")
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- 4: Borda consensus ---
    st.subheader("4️⃣ Консенсусне ранжування (метод Борда)")
    borda = {a: sum(ranks_table[m].get(a, len(alts)) for m in methods_list) for a in alts}
    borda_df = pd.DataFrame({"Бал Борда (менший = кращий)": borda}).sort_values("Бал Борда (менший = кращий)")
    borda_df.index.name = "Альтернатива"
    st.dataframe(borda_df, use_container_width=True)

    # --- 5: Correlation matrices ---
    st.subheader("5️⃣ Кореляція між методами")
    rank_vecs = {m: [ranks_table[m].get(a, len(alts)) for a in alts] for m in methods_list}
    n_m = len(methods_list)
    sp_mat = np.zeros((n_m, n_m))
    kd_mat = np.zeros((n_m, n_m))
    for i, m1 in enumerate(methods_list):
        for j, m2 in enumerate(methods_list):
            if i == j:
                sp_mat[i, j] = 1.0
                kd_mat[i, j] = 1.0
            else:
                rho, _ = spearmanr(rank_vecs[m1], rank_vecs[m2])
                tau, _ = kendalltau(rank_vecs[m1], rank_vecs[m2])
                sp_mat[i, j] = round(float(rho), 3)
                kd_mat[i, j] = round(float(tau), 3)

    col_s, col_k = st.columns(2)
    with col_s:
        st.write("**Кореляція Спірмена**")
        st.dataframe(pd.DataFrame(sp_mat, index=methods_list, columns=methods_list).round(3),
                     use_container_width=True)
    with col_k:
        st.write("**Тау Кендалла**")
        st.dataframe(pd.DataFrame(kd_mat, index=methods_list, columns=methods_list).round(3),
                     use_container_width=True)

    # Heatmap
    fig_heat = px.imshow(
        pd.DataFrame(sp_mat, index=methods_list, columns=methods_list),
        color_continuous_scale="RdBu", zmin=-1, zmax=1,
        title="Теплова карта кореляції Спірмена між методами",
        text_auto=".2f",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- 6: Contradictory alternatives ---
    st.subheader("6️⃣ Суперечливі альтернативи")
    contradictory = []
    for alt in alts:
        alt_ranks = [ranks_table[m].get(alt, len(alts)) for m in methods_list]
        if max(alt_ranks) - min(alt_ranks) > len(alts) / 2:
            contradictory.append(alt)
    if contradictory:
        st.warning(f"Методи дають суперечливі ранги (різниця > n/2) для: **{', '.join(contradictory)}**")
    else:
        st.success("✅ Всі методи узгоджено — суперечливих альтернатив не виявлено.")

    # --- PDF ---
    st.subheader("7️⃣ Звіт PDF")
    tasks_df = load_tasks()
    task_name_cur = tasks_df[tasks_df["id"] == tid]["name"].iloc[0]
    if st.button("📄 Генерувати PDF-звіт"):
        with st.spinner("Генерація PDF…"):
            try:
                pdf_bytes = generate_pdf_report(results, task_name_cur)
                st.download_button(
                    label="⬇️ Завантажити PDF",
                    data=pdf_bytes,
                    file_name=f"mcdm_report_{tid}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Помилка генерації PDF: {e}")

# ============================================================
# 🔄 What-if аналіз
# ============================================================
elif page == "🔄 What-if аналіз":
    st.header("🔄 What-if аналіз та Sensitivity Analysis")
    tid = task_selectbox(key="whatif_task_sel")
    if tid is None:
        st.stop()

    criteria = load_criteria(tid)
    alternatives = load_alternatives(tid)
    evaluations = load_evaluations(tid)

    if criteria.empty or alternatives.empty or evaluations.empty:
        st.warning("Спочатку заповніть задачу (критерії, альтернативи, матриця).")
        st.stop()

    matrix = build_matrix(criteria, alternatives, evaluations)
    if matrix.isnull().any().any():
        st.warning("Матриця оцінок неповна.")
        st.stop()

    tab_whatif, tab_sens = st.tabs(["What-if моделювання", "Аналіз чутливості"])

    # ---- What-if ----
    with tab_whatif:
        st.info("Змініть значення в матриці нижче і натисніть «Перерахувати» для порівняння.")
        edited = st.data_editor(
            matrix.copy(),
            use_container_width=True,
            num_rows="fixed",
            key="whatif_editor",
            column_config={c: st.column_config.NumberColumn(c) for c in criteria["name"].tolist()},
        )

        if st.button("🔄 Перерахувати всі методи після змін", type="primary"):
            with st.spinner("Розрахунок базового та зміненого сценаріїв…"):
                try:
                    base = run_all_mcdm(engine, tid)
                    modified = run_all_mcdm(engine, tid, override_matrix=edited.astype(float))

                    if not base["success"] or not modified["success"]:
                        st.error(base.get("message") or modified.get("message"))
                        st.stop()

                    st.subheader("Порівняння сценаріїв")
                    methods_cmp = list(base["ranks_table"].keys())
                    alts_cmp = base["alternatives"]

                    rows = []
                    for alt in alts_cmp:
                        row = {"Альтернатива": alt}
                        for m in methods_cmp:
                            b_rank = base["ranks_table"][m].get(alt, "-")
                            n_rank = modified["ranks_table"][m].get(alt, "-")
                            if isinstance(b_rank, int) and isinstance(n_rank, int):
                                sym = "↑" if n_rank < b_rank else ("↓" if n_rank > b_rank else "=")
                                row[m] = f"{b_rank}→{n_rank} {sym}"
                            else:
                                row[m] = f"{b_rank}→{n_rank}"
                        rows.append(row)

                    cmp_df = pd.DataFrame(rows).set_index("Альтернатива")
                    st.dataframe(cmp_df, use_container_width=True)

                    st.caption("↑ = покращення рангу, ↓ = погіршення, = = без змін")
                except Exception as e:
                    st.error(f"Помилка розрахунку: {e}")

    # ---- Sensitivity Analysis ----
    with tab_sens:
        st.info("Оберіть критерій та метод для побудови графіку чутливості рангів до зміни ваги.")
        crit_names = criteria["name"].tolist()
        sel_crit = st.selectbox("Критерій для аналізу", crit_names, key="sens_crit")
        sel_method = st.selectbox("Метод MCDM", ["WSM", "WPM", "TOPSIS", "VIKOR", "ELECTRE I", "AHP"],
                                  key="sens_method")

        base_weights = load_weights(tid, criteria)
        crit_idx = crit_names.index(sel_crit)
        crit_types = criteria["type"].values

        st.markdown(f"Поточна вага критерію **{sel_crit}**: `{base_weights[crit_idx]:.4f}`")

        if st.button("📊 Побудувати графік чутливості"):
            with st.spinner("Обчислення 21 сценарію…"):
                try:
                    weight_steps = np.linspace(0.0, 1.0, 21)
                    records = []
                    for w_val in weight_steps:
                        new_w = base_weights.copy()
                        other_sum = 1.0 - base_weights[crit_idx]
                        new_w[crit_idx] = w_val
                        remaining = 1.0 - w_val
                        if other_sum > 1e-10:
                            scale = remaining / other_sum
                            for j in range(len(new_w)):
                                if j != crit_idx:
                                    new_w[j] = base_weights[j] * scale
                        else:
                            n_other = len(new_w) - 1
                            for j in range(len(new_w)):
                                if j != crit_idx:
                                    new_w[j] = remaining / n_other if n_other > 0 else 0.0

                        res = run_methods_only(matrix, new_w, crit_types)
                        ranks_m = res["ranks_table"].get(sel_method, {})
                        for alt in alternatives["name"].tolist():
                            records.append({
                                "Вага критерію": round(float(w_val), 2),
                                "Альтернатива": alt,
                                "Ранг": ranks_m.get(alt, len(alternatives)),
                            })

                    sens_df = pd.DataFrame(records)
                    fig_sens = px.line(
                        sens_df, x="Вага критерію", y="Ранг", color="Альтернатива",
                        markers=True,
                        title=f"Чутливість рангів [{sel_method}] до ваги «{sel_crit}»",
                    )
                    fig_sens.update_yaxes(autorange="reversed", dtick=1,
                                         title="Позиція (1 = найкраща)")
                    fig_sens.update_xaxes(title=f"Вага критерію «{sel_crit}»")
                    st.plotly_chart(fig_sens, use_container_width=True)
                    st.caption("Точки перетину ліній — моменти зміни ранжування альтернатив.")
                except Exception as e:
                    st.error(f"Помилка аналізу чутливості: {e}")

# ============================================================
# 📜 Історія рішень
# ============================================================
elif page == "📜 Історія рішень":
    st.header("📜 Історія прийнятих рішень")
    try:
        with engine.connect() as conn:
            history = pd.read_sql(
                text("""
                    SELECT r.id, t.name AS task_name, r.method_name,
                           r.ranking, r.computed_at
                    FROM mcdm_results r
                    JOIN tasks t ON r.task_id = t.id
                    ORDER BY r.computed_at DESC
                """),
                conn,
            )

        if history.empty:
            st.info("Розрахунків ще немає. Виконайте MCDM-розрахунки.")
        else:
            # Filtering
            tasks_in_history = ["Всі"] + sorted(history["task_name"].unique().tolist())
            filter_task = st.selectbox("Фільтр за задачею", tasks_in_history, key="hist_filter")
            if filter_task != "Всі":
                history = history[history["task_name"] == filter_task]

            # Add top recommendation column
            def top_alt(ranking_str: str) -> str:
                if "→" in str(ranking_str):
                    return str(ranking_str).split("→")[0].strip()
                return str(ranking_str).strip() if pd.notna(ranking_str) else "—"

            history = history.copy()
            history["Рекомендована альтернатива"] = history["ranking"].apply(top_alt)
            history = history.rename(columns={
                "task_name": "Задача",
                "method_name": "Метод",
                "ranking": "Ранжування",
                "computed_at": "Дата розрахунку",
            })
            st.dataframe(
                history[["Задача", "Метод", "Рекомендована альтернатива", "Ранжування", "Дата розрахунку"]],
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Помилка завантаження історії: {e}")

# ============================================================
# Footer
# ============================================================
st.sidebar.caption("Бакалаврська робота • Комп'ютерні науки • Повна версія")
