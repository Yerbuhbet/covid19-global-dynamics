import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import CLEAN_COUNTRIES_PARQUET, FIGURES_DIR, METRICS_DIR

TRAIN_START = "2020-03-01"
TRAIN_END = "2022-06-30"
TEST_START = "2022-07-01"
TEST_END = "2022-10-31"

FORECAST_SERIES = [
    ("GLOBAL", "Global (237 countries)"),
    ("KAZ", "Kazakhstan"),
    ("USA", "United States"),
    ("JPN", "Japan"),
    ("BRA", "Brazil"),
]

LAG_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "rolling_mean_7"]

FORECAST_METRICS_PATH = METRICS_DIR / "forecast_comparison.csv"
FORECAST_PREDS_PATH = METRICS_DIR / "forecast_predictions.csv"
FORECAST_SUMMARY_PATH = METRICS_DIR / "forecast_summary.json"


def prepare_series(df: pd.DataFrame, code: str) -> pd.Series:
    """Извлекает непрерывный дневной ряд new_cases_smoothed для GLOBAL или конкретной страны."""
    sub = df[(df["date"] >= TRAIN_START) & (df["date"] <= TEST_END)]
    if code == "GLOBAL":
        s = sub.groupby("date")["new_cases_smoothed"].sum()
    else:
        s = sub[sub["iso_code"] == code].groupby("date")["new_cases_smoothed"].sum()

    s = s.asfreq("D").astype("float64").interpolate(limit=7).fillna(0.0)
    return s


def build_lag_frame(series: pd.Series) -> pd.DataFrame:
    """Создаёт таблицу признаков строго из прошлых значений (shift >= 1 — без data leakage)."""
    frame = pd.DataFrame({"y": series})
    frame["lag_1"] = frame["y"].shift(1)
    frame["lag_2"] = frame["y"].shift(2)
    frame["lag_3"] = frame["y"].shift(3)
    frame["lag_7"] = frame["y"].shift(7)
    frame["lag_14"] = frame["y"].shift(14)
    # Скользящее среднее считается от lag_1, чтобы текущее значение y[t] не попало в признак
    frame["rolling_mean_7"] = frame["y"].shift(1).rolling(7).mean()
    return frame.dropna()


def evaluate_models_for_series(
    code: str,
    label: str,
    series: pd.Series,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Обучает и сравнивает 4 подхода на одном ряде строго без утечки данных."""
    lag_df = build_lag_frame(series)

    train_df = lag_df.loc[lag_df.index <= TRAIN_END]
    test_df = lag_df.loc[(lag_df.index >= TEST_START) & (lag_df.index <= TEST_END)]

    if len(train_df) < 100 or len(test_df) < 30:
        raise ValueError(f"Недостаточно данных для {code}: train={len(train_df)}, test={len(test_df)}")

    X_train = train_df[LAG_FEATURES].values
    y_train = train_df["y"].values
    X_test = test_df[LAG_FEATURES].values
    y_test = test_df["y"].values
    test_mean = float(np.mean(y_test))

    # Словарь моделей: (название, тип, функция (fit, predict))
    model_specs = [
        (
            "Naive (baseline)",
            "baseline",
            lambda: None,
            lambda _: test_df["lag_1"].values.copy(),
        ),
        (
            "Seasonal Naive 7d (baseline)",
            "baseline",
            lambda: None,
            lambda _: test_df["lag_7"].values.copy(),
        ),
        (
            "Moving Average 7d",
            "model",
            lambda: None,
            lambda _: test_df["rolling_mean_7"].values.copy(),
        ),
        (
            "Linear Regression (AR lags)",
            "model",
            lambda: LinearRegression().fit(X_train, y_train),
            lambda m: np.clip(m.predict(X_test), 0.0, None),
        ),
    ]

    metrics_rows: List[Dict] = []
    pred_records: List[pd.DataFrame] = []

    for model_name, role, fit_fn, pred_fn in model_specs:
        # 1. Чистый замер времени (без tracemalloc)
        t0 = time.perf_counter()
        fitted = fit_fn()
        fit_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        y_pred = pred_fn(fitted)
        pred_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Отдельный замер пиковой памяти
        tracemalloc.start()
        _m = fit_fn()
        _ = pred_fn(_m)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        nmae_pct = (mae / test_mean * 100.0) if test_mean > 0 else np.nan

        metrics_rows.append(
            {
                "series_code": code,
                "series_name": label,
                "model": model_name,
                "role": role,
                "train_days": len(train_df),
                "test_days": len(test_df),
                "test_mean_actual": round(test_mean, 2),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "nmae_pct": round(nmae_pct, 2),
                "fit_ms": round(fit_ms, 3),
                "predict_ms": round(pred_ms, 3),
                "peak_memory_kb": round(peak_bytes / 1024.0, 2),
            }
        )

        pred_records.append(
            pd.DataFrame(
                {
                    "date": test_df.index,
                    "series_code": code,
                    "series_name": label,
                    "model": model_name,
                    "actual": np.round(y_test, 2),
                    "predicted": np.round(y_pred, 2),
                }
            )
        )

    return metrics_rows, pd.concat(pred_records, ignore_index=True)


def plot_forecast_results(metrics_df: pd.DataFrame, preds_df: pd.DataFrame) -> None:
    """Строит сводный график: факт vs модели (Global и Казахстан) + сравнение nMAE."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # Панели 1 и 2: Факт vs Прогноз для GLOBAL и KAZ
    styles = {
        "Naive (baseline)": ("#ff7f0e", "--", 1.4),
        "Seasonal Naive 7d (baseline)": ("#9467bd", ":", 1.4),
        "Moving Average 7d": ("#2ca02c", "-.", 1.4),
        "Linear Regression (AR lags)": ("#d62728", "-", 1.8),
    }

    for ax, code, title in [
        (axes[0], "GLOBAL", "Глобальная заболеваемость (Global): факт против прогнозов на тесте (июль–октябрь 2022)"),
        (axes[1], "KAZ", "Казахстан (KAZ): факт против прогнозов на волне W5 (июль–октябрь 2022)"),
    ]:
        sub = preds_df[preds_df["series_code"] == code]
        actual_sub = sub[sub["model"] == "Naive (baseline)"]
        ax.plot(actual_sub["date"], actual_sub["actual"], color="black", linewidth=2.2, label="Факт (actual)")

        for m_name, (color, ls, lw) in styles.items():
            m_sub = sub[sub["model"] == m_name]
            m_mae = metrics_df[(metrics_df["series_code"] == code) & (metrics_df["model"] == m_name)]["mae"].iloc[0]
            ax.plot(
                m_sub["date"],
                m_sub["predicted"],
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=0.9,
                label=f"{m_name} (MAE={m_mae:,.1f})",
            )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Новые случаи (7d сглаж.)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    # Панель 3: Сравнение нормализованной ошибки nMAE (%) по всем 5 рядам
    pivot_nmae = metrics_df.pivot(index="series_code", columns="model", values="nmae_pct")
    col_order = [
        "Seasonal Naive 7d (baseline)",
        "Moving Average 7d",
        "Naive (baseline)",
        "Linear Regression (AR lags)",
    ]
    pivot_nmae = pivot_nmae[[c for c in col_order if c in pivot_nmae.columns]]
    pivot_nmae.plot(kind="bar", ax=axes[2], color=["#9467bd", "#2ca02c", "#ff7f0e", "#d62728"], width=0.75)
    axes[2].set_title("Сравнение моделей по нормализованной ошибке nMAE (% от среднего факта — меньше = лучше)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("nMAE (%)")
    axes[2].set_xlabel("Ряд / Страна")
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].legend(loc="upper right", fontsize=8, ncol=2)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_fig = FIGURES_DIR / "fig6_forecast_comparison.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"[Forecast] График сравнения моделей сохранён -> {out_fig.name}")


def run_forecasting_pipeline() -> None:
    """Точка входа Шага 4: честное прогнозирование без data leakage по 5 рядам."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_parquet(CLEAN_COUNTRIES_PARQUET)
        df["date"] = pd.to_datetime(df["date"])

        all_metrics: List[Dict] = []
        all_preds: List[pd.DataFrame] = []

        for code, label in FORECAST_SERIES:
            series = prepare_series(df, code)
            m_rows, p_df = evaluate_models_for_series(code, label, series)
            all_metrics.extend(m_rows)
            all_preds.append(p_df)

        metrics_df = pd.DataFrame(all_metrics)
        preds_df = pd.concat(all_preds, ignore_index=True)

        metrics_df.to_csv(FORECAST_METRICS_PATH, index=False)
        preds_df.to_csv(FORECAST_PREDS_PATH, index=False)
        plot_forecast_results(metrics_df, preds_df)

        # Средний рейтинг моделей по nMAE (%) по всем 5 рядам
        avg_ranking = (
            metrics_df.groupby("model", as_index=False)
            .agg(
                mean_nmae_pct=("nmae_pct", "mean"),
                mean_fit_ms=("fit_ms", "mean"),
                mean_predict_ms=("predict_ms", "mean"),
                mean_peak_kb=("peak_memory_kb", "mean"),
            )
            .sort_values("mean_nmae_pct")
            .round(3)
        )

        best_model = str(avg_ranking.iloc[0]["model"])
        summary = {
            "split": {"train": [TRAIN_START, TRAIN_END], "test": [TEST_START, TEST_END]},
            "leakage_free": True,
            "best_model_by_mean_nmae": best_model,
            "avg_ranking": avg_ranking.to_dict(orient="records"),
        }
        with open(FORECAST_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n--- Детальная таблица прогнозирования (без data leakage) ---")
        cols_show = ["series_code", "model", "mae", "rmse", "nmae_pct", "fit_ms", "peak_memory_kb"]
        print(metrics_df[cols_show].to_string(index=False))

        print("\n--- Итоговый рейтинг моделей (средний nMAE % по всем 5 рядам) ---")
        print(avg_ranking.to_string(index=False))

    except FileNotFoundError as exc:
        print(f"[ОШИБКА ФАЙЛА] {exc}")
    except Exception as exc:
        print(f"[НЕОЖИДАННАЯ ОШИБКА] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_forecasting_pipeline()
