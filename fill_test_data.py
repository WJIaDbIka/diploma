import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///mcdm_decisions.db")

print("🧹 Очищення старих даних...")
with engine.connect() as conn:
    conn.execute(text("DELETE FROM tasks WHERE name LIKE '%постачальника%'"))
    conn.commit()

print("📋 Створення задачі...")
with engine.connect() as conn:
    conn.execute(text("""
        INSERT INTO tasks (name, description) 
        VALUES ('Вибір постачальника промислового обладнання', 
                'Реалістична тестова задача для перевірки MCDM-системи')
    """))
    task_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
    conn.commit()

print(f"✅ Задача створена (ID = {task_id})")

# Критерії
criteria = [
    ("Вартість обладнання, тис. грн", "cost", "тис. грн"),
    ("Якість обладнання (1-10)", "benefit", "бали"),
    ("Термін поставки, днів", "cost", "днів"),
    ("Гарантійний термін, місяців", "benefit", "місяців"),
    ("Післяпродажний сервіс (1-10)", "benefit", "бали"),
    ("Екологічність (1-10)", "benefit", "бали")
]

with engine.connect() as conn:
    for name, ctype, unit in criteria:
        conn.execute(text("INSERT INTO criteria (task_id, name, type, unit) VALUES (:t, :n, :ty, :u)"),
                     {"t": task_id, "n": name, "ty": ctype, "u": unit})
    conn.commit()

print("✅ Критерії додано")

# Альтернативи
alts = ["Постачальник A (Німеччина)", "Постачальник B (Туреччина)", 
        "Постачальник C (Китай)", "Постачальник D (Польща)"]

with engine.connect() as conn:
    for name in alts:
        conn.execute(text("INSERT INTO alternatives (task_id, name) VALUES (:t, :n)"),
                     {"t": task_id, "n": name})
    conn.commit()

print("✅ Альтернативи додано")

# Матриця оцінок
matrix_data = {
    "Постачальник A (Німеччина)": [1250, 9.5, 45, 36, 9.8, 9.2],
    "Постачальник B (Туреччина)": [980, 8.2, 60, 24, 8.5, 7.8],
    "Постачальник C (Китай)": [720, 7.1, 90, 18, 6.5, 6.0],
    "Постачальник D (Польща)": [1100, 8.8, 55, 30, 8.9, 8.5]
}

with engine.connect() as conn:
    alt_df = pd.read_sql(f"SELECT id, name FROM alternatives WHERE task_id = {task_id}", conn)
    crit_df = pd.read_sql(f"SELECT id FROM criteria WHERE task_id = {task_id}", conn)
    
    for alt_name, values in matrix_data.items():
        alt_id = int(alt_df[alt_df['name'] == alt_name]['id'].iloc[0])
        for i, value in enumerate(values):
            crit_id = int(crit_df.iloc[i]['id'])
            conn.execute(text("""
                INSERT INTO evaluations (alternative_id, criterion_id, value)
                VALUES (:a, :c, :v)
            """), {"a": alt_id, "c": crit_id, "v": float(value)})
    conn.commit()

print("✅ Матриця оцінок повністю заповнена!")
print("\n🎉 Готово! Перезапустіть Streamlit і перевірте вкладку «Матриця оцінок».")