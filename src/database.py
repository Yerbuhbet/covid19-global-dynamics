import sqlite3
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import CLEAN_COUNTRIES_PARQUET, DB_PATH, OUTPUTS_DIR, SQL_DIR

SCHEMA_PATH = SQL_DIR / "schema.sql"
QUERIES_PATH = SQL_DIR / "queries.sql"
SQL_OUTPUT_DIR = OUTPUTS_DIR / "sql"

LOCATION_COLS = [
    "iso_code", "continent", "location", "population",
    "population_density", "median_age", "gdp_per_capita",
    "life_expectancy", "human_development_index",
]

EPIDEMIOLOGY_COLS = [
    "iso_code", "date", "total_cases", "new_cases", "new_cases_smoothed",
    "total_deaths", "new_deaths", "new_deaths_smoothed",
    "total_cases_per_million", "total_deaths_per_million",
    "new_cases_per_million", "new_deaths_per_million", "reproduction_rate",
]

VACCINATION_COLS = [
    "iso_code", "date", "total_vaccinations", "people_vaccinated",
    "people_fully_vaccinated", "total_boosters",
    "new_vaccinations", "new_vaccinations_smoothed",
    "people_fully_vaccinated_per_hundred",
]

HOSPITAL_POLICY_COLS = [
    "iso_code", "date", "icu_patients", "hosp_patients", "stringency_index",
]


def build_sqlite_database(
    parquet_path: Path = CLEAN_COUNTRIES_PARQUET,
    db_path: Path = DB_PATH,
    schema_path: Path = SCHEMA_PATH,
) -> Dict[str, int]:
    """Создаёт нормализованную БД SQLite (4 таблицы) из очищенного Parquet."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet не найден: {parquet_path}. Сначала запустите src.data_loader.")
    if not schema_path.exists():
        raise FileNotFoundError(f"Файл схемы SQL не найден: {schema_path}.")

    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path)

    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    loc_df = df[LOCATION_COLS].groupby("iso_code", as_index=False).first()

    epi_metric_cols = [c for c in EPIDEMIOLOGY_COLS if c not in ("iso_code", "date")]
    epi_df = df[EPIDEMIOLOGY_COLS].dropna(subset=epi_metric_cols, how="all")

    vacc_metric_cols = [c for c in VACCINATION_COLS if c not in ("iso_code", "date")]
    vacc_df = df[VACCINATION_COLS].dropna(subset=vacc_metric_cols, how="all")

    hosp_metric_cols = [c for c in HOSPITAL_POLICY_COLS if c not in ("iso_code", "date")]
    hosp_df = df[HOSPITAL_POLICY_COLS].dropna(subset=hosp_metric_cols, how="all")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")

    table_counts: Dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        loc_df.to_sql("locations", conn, if_exists="append", index=False, chunksize=10_000)
        epi_df.to_sql("epidemiology", conn, if_exists="append", index=False, chunksize=50_000)
        vacc_df.to_sql("vaccinations", conn, if_exists="append", index=False, chunksize=50_000)
        hosp_df.to_sql("hospital_policy", conn, if_exists="append", index=False, chunksize=50_000)

        for tbl in ("locations", "epidemiology", "vaccinations", "hospital_policy"):
            cur = conn.execute(f"SELECT COUNT(*) FROM {tbl}")
            table_counts[tbl] = int(cur.fetchone()[0])

    elapsed = round(time.perf_counter() - t0, 2)
    db_size_mb = round(db_path.stat().st_size / (1024 ** 2), 2)
    print(f"[DB] База создана за {elapsed}s ({db_size_mb} МБ): {db_path}")
    for tbl, cnt in table_counts.items():
        print(f"     - {tbl}: {cnt:,} строк")
    return table_counts


def parse_named_queries(queries_path: Path = QUERIES_PATH) -> Dict[str, str]:
    """Разбирает файл sql/queries.sql на словарь {имя_запроса: SQL_текст}."""
    if not queries_path.exists():
        raise FileNotFoundError(f"Файл SQL-запросов не найден: {queries_path}")

    queries: Dict[str, str] = {}
    current_name = None
    buffer = []

    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("-- name:"):
            if current_name and buffer:
                queries[current_name] = "\n".join(buffer).strip()
            current_name = line.split("-- name:", 1)[1].strip()
            buffer = []
        else:
            if current_name is not None:
                buffer.append(line)

    if current_name and buffer:
        queries[current_name] = "\n".join(buffer).strip()

    return queries


def execute_all_queries(
    db_path: Path = DB_PATH,
    queries_path: Path = QUERIES_PATH,
    output_dir: Path = SQL_OUTPUT_DIR,
) -> Dict[str, pd.DataFrame]:
    """Выполняет все SQL-запросы, загружает результаты в Pandas и сохраняет в CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = parse_named_queries(queries_path)
    results: Dict[str, pd.DataFrame] = {}

    with sqlite3.connect(db_path) as conn:
        for name, sql_text in queries.items():
            t0 = time.perf_counter()
            res_df = pd.read_sql_query(sql_text, conn)
            dt = round(time.perf_counter() - t0, 3)
            out_file = output_dir / f"{name}.csv"
            res_df.to_csv(out_file, index=False)
            results[name] = res_df
            print(f"[SQL] {name}: {len(res_df):,} строк x {res_df.shape[1]} кол. ({dt}s) -> {out_file.name}")

    return results


def run_database_pipeline() -> None:
    """Точка входа Шага 2: сборка SQLite и выполнение всех 9 SQL-запросов."""
    try:
        build_sqlite_database()
        results = execute_all_queries()

        print("\n--- Превью q01 (Сводка по континентам) ---")
        print(results["q01_continent_summary"].to_string(index=False))

        print("\n--- Превью q07 (Смертность до и после 50% вакцинации) ---")
        print(results["q07_mortality_before_after_50pct_vacc"].to_string(index=False))

        print("\n--- Превью q09 (Казахстан по годам + YoY %) ---")
        print(results["q09_kazakhstan_yearly_yoy"].to_string(index=False))

    except FileNotFoundError as exc:
        print(f"[ОШИБКА ФАЙЛА] {exc}")
    except sqlite3.Error as exc:
        print(f"[ОШИБКА SQLITE] {exc}")
    except Exception as exc:
        print(f"[НЕОЖИДАННАЯ ОШИБКА] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_database_pipeline()
