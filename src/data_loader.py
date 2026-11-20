import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CHUNK_SIZE,
    CLEAN_AGGREGATES_PARQUET,
    CLEAN_COUNTRIES_PARQUET,
    COLUMN_RENAME_MAP,
    EDA_AUDIT_PATH,
    IQR_TARGET_COLUMNS,
    METRICS_DIR,
    NON_NEGATIVE_COLUMNS,
    OUTLIERS_SUMMARY_PATH,
    PROCESSED_DIR,
    RAW_CSV_PATH,
    STORAGE_BENCHMARK_PATH,
)


def load_raw_in_chunks(csv_path: Path, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Читает большой CSV чанками и стандартизирует названия ключевых колонок."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Исходный файл не найден: {csv_path}. Скачайте compact.csv из OWID."
        )

    chunks: List[pd.DataFrame] = []
    chunk_count = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunk = chunk.rename(columns=COLUMN_RENAME_MAP)
        chunks.append(chunk)
        chunk_count += 1

    if not chunks:
        raise ValueError(f"Файл {csv_path} пуст — не удалось прочитать ни одной строки.")

    df = pd.concat(chunks, ignore_index=True)
    print(f"[1/6] Прочитано чанков: {chunk_count} | Строк: {len(df):,} | Колонок: {df.shape[1]}")
    return df


def clean_and_split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Очищает данные и разделяет реальные страны и агрегаты OWID."""
    report: Dict = {}
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates > 0:
        df = df.dropna(subset=["date"])
    report["invalid_dates_dropped"] = invalid_dates

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["location", "date"], keep="last")
    report["duplicates_dropped"] = before_dedup - len(df)

    negative_fixes: Dict[str, int] = {}
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            neg_mask = df[col] < 0
            neg_count = int(neg_mask.sum())
            if neg_count > 0:
                df.loc[neg_mask, col] = np.nan
                negative_fixes[col] = neg_count
    report["negative_values_set_to_nan"] = negative_fixes

    iso_str = df["iso_code"].fillna("").astype(str) if "iso_code" in df.columns else pd.Series("", index=df.index)
    if "continent" in df.columns:
        is_aggregate = df["continent"].isna() | iso_str.str.startswith("OWID_") | (iso_str == "")
    else:
        is_aggregate = iso_str.str.startswith("OWID_") | (iso_str == "")

    aggregates_df = df[is_aggregate].copy()
    countries_df = df[~is_aggregate].copy()

    countries_df = countries_df.sort_values(["iso_code", "date"]).reset_index(drop=True)
    aggregates_df = aggregates_df.sort_values(["location", "date"]).reset_index(drop=True)

    report["total_rows_after_dedup"] = len(df)
    report["country_rows"] = len(countries_df)
    report["aggregate_rows"] = len(aggregates_df)
    report["unique_countries"] = int(countries_df["iso_code"].nunique())
    report["unique_aggregates"] = int(aggregates_df["location"].nunique())
    report["date_min"] = str(countries_df["date"].min().date())
    report["date_max"] = str(countries_df["date"].max().date())

    print(
        f"[2/6] Разделение: страны = {len(countries_df):,} строк "
        f"({report['unique_countries']} стран), "
        f"агрегаты = {len(aggregates_df):,} строк ({report['unique_aggregates']} групп)"
    )
    return countries_df, aggregates_df, report


def optimize_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Оптимизирует типы данных (float64 -> float32, строки -> category) и замеряет экономию RAM."""
    mem_before_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = df_opt[col].astype("float32")

    for col in df_opt.select_dtypes(include=["int64"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    for col in df_opt.select_dtypes(include=["object", "string"]).columns:
        if df_opt[col].nunique(dropna=True) / max(len(df_opt), 1) < 0.5:
            df_opt[col] = df_opt[col].astype("category")

    mem_after_mb = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    saved_pct = (1 - mem_after_mb / mem_before_mb) * 100 if mem_before_mb > 0 else 0.0

    stats = {
        "memory_before_mb": round(float(mem_before_mb), 2),
        "memory_after_mb": round(float(mem_after_mb), 2),
        "memory_saved_pct": round(float(saved_pct), 2),
    }
    print(
        f"[3/6] Оптимизация RAM: {stats['memory_before_mb']} МБ -> "
        f"{stats['memory_after_mb']} МБ (экономия {stats['memory_saved_pct']}%)"
    )
    return df_opt, stats


def audit_data_quality(df: pd.DataFrame, cleaning_report: Dict, mem_stats: Dict) -> Dict:
    """Формирует EDA-аудит: типы признаков, пропуски и описательную статистику."""
    missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    dtypes_summary = df.dtypes.astype(str).value_counts().to_dict()

    key_numeric = [
        c for c in [
            "new_cases",
            "new_deaths",
            "total_cases",
            "total_deaths",
            "new_cases_per_million",
            "new_deaths_per_million",
            "people_fully_vaccinated_per_hundred",
            "population",
            "gdp_per_capita",
            "stringency_index",
        ]
        if c in df.columns
    ]

    desc_stats = {}
    for col in key_numeric:
        s = df[col].dropna()
        if len(s) > 0:
            desc_stats[col] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "median": round(float(s.median()), 2),
                "max": round(float(s.max()), 2),
                "missing_pct": float(missing_pct[col]),
            }

    audit = {
        "cleaning_report": cleaning_report,
        "memory_optimization": mem_stats,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes_counts": dtypes_summary,
        "overall_missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "top_10_missing_columns": missing_pct.head(10).to_dict(),
        "descriptive_stats": desc_stats,
    }
    print(f"[4/6] EDA-аудит собран: всего пропусков {audit['overall_missing_pct']}%, признаков {df.shape[1]}")
    return audit


def detect_outliers_iqr(df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    """Обнаруживает выбросы методом IQR (для дневных счётчиков оценивает значения > 0)."""
    rows: List[Dict] = []

    for col in target_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if col.startswith("new_"):
            series = series[series > 0]
            scope = "positive_only (>0)"
        else:
            scope = "all_non_null"

        if len(series) < 10:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers_mask = (series < lower) | (series > upper)
        outlier_count = int(outliers_mask.sum())
        outlier_pct = round(outlier_count / len(series) * 100, 2)

        rows.append(
            {
                "column": col,
                "scope": scope,
                "n_evaluated": int(len(series)),
                "q1": round(q1, 2),
                "median": round(float(series.median()), 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "max_value": round(float(series.max()), 2),
            }
        )

    result_df = pd.DataFrame(rows)
    print(f"[5/6] Анализ выбросов IQR завершён по {len(result_df)} колонкам")
    return result_df


def benchmark_csv_vs_parquet(csv_path: Path, parquet_path: Path) -> Dict:
    """Сравнивает время чтения (все колонки и аналитический срез), RAM и размер: CSV vs Parquet."""
    subset_cols = [
        "iso_code", "continent", "location", "date",
        "new_cases", "new_deaths", "new_cases_per_million",
        "people_fully_vaccinated_per_hundred", "population", "stringency_index",
    ]
    csv_subset_cols = [
        ("code" if c == "iso_code" else "country" if c == "location" else c)
        for c in subset_cols
    ]

    _ = pd.read_parquet(parquet_path, columns=["iso_code"])

    t0 = time.perf_counter()
    df_csv = pd.read_csv(csv_path, low_memory=False)
    csv_full_time = time.perf_counter() - t0
    csv_ram_mb = df_csv.memory_usage(deep=True).sum() / (1024 ** 2)
    csv_rows = len(df_csv)
    del df_csv

    t0 = time.perf_counter()
    df_pq = pd.read_parquet(parquet_path)
    pq_full_time = time.perf_counter() - t0
    pq_ram_mb = df_pq.memory_usage(deep=True).sum() / (1024 ** 2)
    pq_rows = len(df_pq)
    del df_pq

    t0 = time.perf_counter()
    _ = pd.read_csv(csv_path, usecols=csv_subset_cols, low_memory=False)
    csv_sub_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pd.read_parquet(parquet_path, columns=subset_cols)
    pq_sub_time = time.perf_counter() - t0

    csv_size_mb = csv_path.stat().st_size / (1024 ** 2)
    pq_size_mb = parquet_path.stat().st_size / (1024 ** 2)

    result = {
        "csv": {
            "rows": csv_rows,
            "full_read_seconds": round(csv_full_time, 3),
            "subset_10cols_seconds": round(csv_sub_time, 3),
            "ram_mb": round(float(csv_ram_mb), 2),
            "file_size_mb": round(csv_size_mb, 2),
        },
        "parquet": {
            "rows": pq_rows,
            "full_read_seconds": round(pq_full_time, 3),
            "subset_10cols_seconds": round(pq_sub_time, 3),
            "ram_mb": round(float(pq_ram_mb), 2),
            "file_size_mb": round(pq_size_mb, 2),
        },
        "full_read_speedup_x": round(csv_full_time / max(pq_full_time, 1e-6), 2),
        "subset_read_speedup_x": round(csv_sub_time / max(pq_sub_time, 1e-6), 2),
        "ram_reduction_pct": round((1 - pq_ram_mb / csv_ram_mb) * 100, 2),
        "size_reduction_x": round(csv_size_mb / max(pq_size_mb, 1e-6), 2),
    }

    print(
        f"[6/6] Бенчмарк CSV vs Parquet: "
        f"все колонки {result['csv']['full_read_seconds']}s -> {result['parquet']['full_read_seconds']}s "
        f"({result['full_read_speedup_x']}x) | "
        f"10 колонок {result['csv']['subset_10cols_seconds']}s -> {result['parquet']['subset_10cols_seconds']}s "
        f"({result['subset_read_speedup_x']}x) | "
        f"RAM {result['csv']['ram_mb']} -> {result['parquet']['ram_mb']} МБ | "
        f"диск {result['csv']['file_size_mb']} -> {result['parquet']['file_size_mb']} МБ ({result['size_reduction_x']}x)"
    )
    return result


def run_data_pipeline() -> None:
    """Полный запуск Шага 1: загрузка, очистка, Parquet, EDA-аудит, IQR, бенчмарк."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw_df = load_raw_in_chunks(RAW_CSV_PATH)
        countries_df, aggregates_df, cleaning_report = clean_and_split_data(raw_df)
        del raw_df

        countries_opt, mem_stats = optimize_dtypes(countries_df)
        aggregates_opt, _ = optimize_dtypes(aggregates_df)

        countries_opt.to_parquet(CLEAN_COUNTRIES_PARQUET, index=False)
        aggregates_opt.to_parquet(CLEAN_AGGREGATES_PARQUET, index=False)

        eda_audit = audit_data_quality(countries_opt, cleaning_report, mem_stats)
        outliers_df = detect_outliers_iqr(countries_opt, IQR_TARGET_COLUMNS)

        with open(EDA_AUDIT_PATH, "w", encoding="utf-8") as f:
            json.dump(eda_audit, f, ensure_ascii=False, indent=2)
        outliers_df.to_csv(OUTLIERS_SUMMARY_PATH, index=False)

        benchmark = benchmark_csv_vs_parquet(RAW_CSV_PATH, CLEAN_COUNTRIES_PARQUET)
        with open(STORAGE_BENCHMARK_PATH, "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=2)

        print("\n--- ГОТОВО! Таблица выбросов (IQR) ---")
        print(outliers_df[["column", "scope", "median", "upper_bound", "outlier_count", "outlier_pct"]].to_string(index=False))

    except FileNotFoundError as exc:
        print(f"[ОШИБКА ФАЙЛА] {exc}")
    except (ValueError, KeyError) as exc:
        print(f"[ОШИБКА ДАННЫХ] {exc}")
    except Exception as exc:
        print(f"[НЕОЖИДАННАЯ ОШИБКА] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_data_pipeline()
