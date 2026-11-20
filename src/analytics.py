import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import (
    CLEAN_COUNTRIES_PARQUET,
    FIGURES_DIR,
    METRICS_DIR,
)

ACTIVE_START = "2020-03-01"
ACTIVE_END = "2023-12-31"
FOCUS_COUNTRIES = ["KAZ", "USA", "DEU", "JPN", "BRA"]

SKEWNESS_REPORT_PATH = METRICS_DIR / "skewness_report.csv"
WEEKLY_TS_PATH = METRICS_DIR / "global_weekly_timeseries.csv"
MONTHLY_TS_PATH = METRICS_DIR / "global_monthly_timeseries.csv"
CORR_MATRIX_PATH = METRICS_DIR / "correlation_matrix.csv"
WAVES_PATH = METRICS_DIR / "waves_detected.csv"
ANALYTICS_SUMMARY_PATH = METRICS_DIR / "analytics_summary.json"


def load_clean_countries(parquet_path: Path = CLEAN_COUNTRIES_PARQUET) -> pd.DataFrame:
    """Загружает очищенный Parquet по странам."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Файл не найден: {parquet_path}. Сначала запустите src.data_loader.")
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def analyze_skewness_and_plot_boxplots(df: pd.DataFrame) -> pd.DataFrame:
    """Считает mean, median, skewness и строит boxplot выбросов (IQR)."""
    metrics = [
        ("new_cases", "Новые случаи (в день, > 0)", True, True),
        ("new_deaths", "Новые смерти (в день, > 0)", True, True),
        ("new_cases_per_million", "Случаи на 1 млн (в день, > 0)", True, True),
        ("people_fully_vaccinated_per_hundred", "Полная вакцинация (%)", False, False),
    ]

    skew_rows: List[Dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes_flat = axes.flatten()

    for ax, (col, title, pos_only, use_log) in zip(axes_flat, metrics):
        s = df[col].dropna().astype("float64")
        if pos_only:
            s = s[s > 0]

        mean_val = float(s.mean())
        med_val = float(s.median())
        skew_val = float(s.skew())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        upper_iqr = q3 + 1.5 * (q3 - q1)

        skew_rows.append(
            {
                "column": col,
                "mean": round(mean_val, 2),
                "median": round(med_val, 2),
                "mean_to_median_ratio": round(mean_val / max(med_val, 1e-6), 2),
                "skewness": round(skew_val, 2),
                "upper_iqr_bound": round(upper_iqr, 2),
            }
        )

        sns.boxplot(x=s, ax=ax, color="#4C72B0", fliersize=2, linewidth=1.2)
        if use_log:
            ax.set_xscale("log")
            ax.set_xlabel(f"{col} (лог. шкала)")
        else:
            ax.set_xlabel(col)
        ax.set_title(f"{title}\nmedian={med_val:.1f}, mean={mean_val:.1f}, skew={skew_val:.2f}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Распределение и выбросы (IQR / Boxplot) ключевых метрик COVID-19", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_fig = FIGURES_DIR / "fig1_outliers_boxplot.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    skew_df = pd.DataFrame(skew_rows)
    skew_df.to_csv(SKEWNESS_REPORT_PATH, index=False)
    print(f"[1/5] Boxplot и асимметрия сохранены -> {out_fig.name}")
    return skew_df


def analyze_time_series_and_decompose(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Строит глобальный ряд, resample (W, ME), скользящие (7/14/30d) и сезонную декомпозицию."""
    active = df[(df["date"] >= ACTIVE_START) & (df["date"] <= ACTIVE_END)]

    daily = (
        active.groupby("date", as_index=True)
        .agg(
            new_cases=("new_cases", "sum"),
            new_deaths=("new_deaths", "sum"),
        )
        .asfreq("D")
        .fillna(0.0)
        .astype("float64")
    )

    daily["ma_7d"] = daily["new_cases"].rolling(7, min_periods=1).mean()
    daily["ma_14d"] = daily["new_cases"].rolling(14, min_periods=1).mean()
    daily["ma_30d"] = daily["new_cases"].rolling(30, min_periods=1).mean()

    weekly = daily[["new_cases", "new_deaths"]].resample("W").sum()
    weekly["ma_4w"] = weekly["new_cases"].rolling(4, min_periods=1).mean()
    weekly.to_csv(WEEKLY_TS_PATH)

    monthly = daily[["new_cases", "new_deaths"]].resample("ME").sum()
    monthly.to_csv(MONTHLY_TS_PATH)

    # Сезонная декомпозиция на недельном ряде (period=52 недели = 1 год)
    decomp = seasonal_decompose(weekly["new_cases"], model="additive", period=52)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(weekly.index, weekly["new_cases"] / 1e6, color="#1f77b4", label="Недельные случаи (млн)")
    axes[0].plot(weekly.index, weekly["ma_4w"] / 1e6, color="#d62728", linewidth=1.8, label="4-нед. скользящее среднее")
    axes[0].set_ylabel("Млн случаев")
    axes[0].set_title("Глобальная динамика и сезонная декомпозиция COVID-19 (2020–2023, шаг = 1 неделя, period=52)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(decomp.trend.index, decomp.trend / 1e6, color="#2ca02c")
    axes[1].set_ylabel("Тренд (млн)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(decomp.seasonal.index, decomp.seasonal / 1e6, color="#ff7f0e")
    axes[2].set_ylabel("Сезонность (млн)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(decomp.resid.index, decomp.resid / 1e6, color="#7f7f7f", linewidth=0.9)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Остатки (млн)")
    axes[3].set_xlabel("Дата")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    out_fig = FIGURES_DIR / "fig2_seasonal_decomposition.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    print(f"[2/5] Временные ряды (W={len(weekly)}, ME={len(monthly)}) и декомпозиция -> {out_fig.name}")
    return weekly, monthly


def compute_correlations_and_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Считает корреляции на уровне стран (население >= 500 тыс.) и строит heatmap."""
    active_policy = df[(df["date"] >= "2020-03-01") & (df["date"] <= "2022-12-31")]
    policy_mean = (
        active_policy.groupby("iso_code", observed=True)["stringency_index"]
        .mean()
        .rename("avg_stringency_2020_2022")
    )

    country_agg = (
        df.groupby("iso_code", observed=True)
        .agg(
            population=("population", "first"),
            total_cases_pm=("total_cases_per_million", "max"),
            total_deaths_pm=("total_deaths_per_million", "max"),
            full_vacc_pct=("people_fully_vaccinated_per_hundred", "max"),
            gdp_per_capita=("gdp_per_capita", "first"),
            median_age=("median_age", "first"),
            hospital_beds=("hospital_beds_per_thousand", "first"),
            life_expectancy=("life_expectancy", "first"),
            pop_density=("population_density", "first"),
        )
        .join(policy_mean)
    )

    country_agg = country_agg[country_agg["population"] >= 500_000].copy()
    country_agg["cfr_percent"] = (
        country_agg["total_deaths_pm"] / country_agg["total_cases_pm"].replace(0, np.nan) * 100.0
    )

    corr_cols = [
        "total_cases_pm",
        "total_deaths_pm",
        "cfr_percent",
        "full_vacc_pct",
        "avg_stringency_2020_2022",
        "gdp_per_capita",
        "median_age",
        "hospital_beds",
        "pop_density",
        "life_expectancy",
    ]
    pretty_names = {
        "total_cases_pm": "Случаи / млн",
        "total_deaths_pm": "Смерти / млн",
        "cfr_percent": "Летальность (CFR %)",
        "full_vacc_pct": "Вакцинация (%)",
        "avg_stringency_2020_2022": "Жёсткость мер",
        "gdp_per_capita": "ВВП на душу",
        "median_age": "Медианный возраст",
        "hospital_beds": "Койки / 1000 чел.",
        "pop_density": "Плотность нас.",
        "life_expectancy": "Ожид. жизнь",
    }

    valid_corr_cols = [c for c in corr_cols if country_agg[c].notna().sum() >= 20]
    corr_df = country_agg[valid_corr_cols].corr(method="pearson").round(3)
    corr_df.to_csv(CORR_MATRIX_PATH)

    plot_corr = corr_df.rename(index=pretty_names, columns=pretty_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(plot_corr, dtype=bool), k=1)
    sns.heatmap(
        plot_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        f"Матрица корреляций (Пирсон, уровень стран, n={len(country_agg)} стран с населением >= 500 тыс.)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    out_fig = FIGURES_DIR / "fig3_correlation_heatmap.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    print(f"[3/5] Корреляции (n={len(country_agg)} стран) и heatmap -> {out_fig.name}")
    return corr_df


def detect_covid_waves(df: pd.DataFrame) -> pd.DataFrame:
    """Детектирует волны заболеваемости по всем странам (2020–2023) через scipy.signal.find_peaks."""
    active = df[(df["date"] >= ACTIVE_START) & (df["date"] <= ACTIVE_END)].copy()
    wave_records: List[Dict] = []

    for iso, grp in active.groupby("iso_code", observed=True):
        grp = grp.sort_values("date")
        location = str(grp["location"].iloc[0])
        continent = str(grp["continent"].iloc[0])

        # Берём сглаженный ряд на миллион, при пропусках считаем 7-дневное скользящее сами
        series = grp.set_index("date")["new_cases_smoothed_per_million"].astype("float64")
        if series.notna().sum() < 90:
            raw_pm = grp.set_index("date")["new_cases_per_million"].astype("float64")
            series = raw_pm.rolling(7, min_periods=1).mean()

        series = series.asfreq("D").interpolate(limit=14).fillna(0.0)
        values = series.values
        max_val = float(values.max())
        if max_val < 20.0:
            continue

        # Пороги: минимум 45 дней между пиками, заметная высота и prominence
        min_height = max(20.0, 0.10 * max_val)
        min_prominence = max(15.0, 0.12 * max_val)
        peaks, props = find_peaks(values, distance=45, height=min_height, prominence=min_prominence)

        for idx, p in enumerate(peaks, start=1):
            wave_records.append(
                {
                    "iso_code": str(iso),
                    "location": location,
                    "continent": continent,
                    "wave_number": idx,
                    "peak_date": str(series.index[p].date()),
                    "peak_cases_pm_smoothed": round(float(values[p]), 2),
                    "prominence": round(float(props["prominences"][idx - 1]), 2),
                }
            )

    waves_df = pd.DataFrame(wave_records)
    waves_df.to_csv(WAVES_PATH, index=False)
    print(
        f"[4/5] Детекция волн завершена: {len(waves_df):,} волн в "
        f"{waves_df['iso_code'].nunique()} странах -> {WAVES_PATH.name}"
    )
    return waves_df


def plot_waves_static_and_interactive(df: pd.DataFrame, waves_df: pd.DataFrame) -> None:
    """Строит статический график (Matplotlib) и интерактивный график волн (Plotly HTML)."""
    active = df[
        (df["iso_code"].isin(FOCUS_COUNTRIES))
        & (df["date"] >= ACTIVE_START)
        & (df["date"] <= ACTIVE_END)
    ].copy()

    # 1. Статический график (Matplotlib) — 5 панелей по странам
    fig, axes = plt.subplots(len(FOCUS_COUNTRIES), 1, figsize=(13, 12), sharex=True)
    for ax, iso in zip(axes, FOCUS_COUNTRIES):
        sub = active[active["iso_code"] == iso].sort_values("date")
        loc_name = str(sub["location"].iloc[0]) if len(sub) > 0 else iso
        ax.plot(sub["date"], sub["new_cases_smoothed_per_million"], color="#1f77b4", linewidth=1.4, label="7d сглаж. случаи / млн")

        sub_waves = waves_df[waves_df["iso_code"] == iso]
        if len(sub_waves) > 0:
            w_dates = pd.to_datetime(sub_waves["peak_date"])
            ax.scatter(w_dates, sub_waves["peak_cases_pm_smoothed"], color="#d62728", s=45, zorder=5, label=f"Пики волн ({len(sub_waves)})")
            for _, row in sub_waves.iterrows():
                ax.annotate(
                    f"W{row['wave_number']}",
                    (pd.to_datetime(row["peak_date"]), row["peak_cases_pm_smoothed"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    color="#b00020",
                )

        y_top = float(sub["new_cases_smoothed_per_million"].max()) if len(sub) > 0 else 1.0
        if y_top > 0:
            ax.set_ylim(0, y_top * 1.18)
        ax.set_ylabel("Случаи / млн")
        ax.set_title(f"{loc_name} ({iso}) — обнаружено волн: {len(sub_waves)}", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Дата")
    fig.suptitle("Детекция волн COVID-19 по ключевым странам (2020–2023)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    static_out = FIGURES_DIR / "fig4_waves_comparison.png"
    fig.savefig(static_out, dpi=150)
    plt.close(fig)

    # 2. Интерактивный график (Plotly) — наложение стран + кликабельная легенда + range slider
    pfig = go.Figure()
    colors = {"KAZ": "#1f77b4", "USA": "#d62728", "DEU": "#2ca02c", "JPN": "#9467bd", "BRA": "#ff7f0e"}

    for iso in FOCUS_COUNTRIES:
        sub = active[active["iso_code"] == iso].sort_values("date")
        if sub.empty:
            continue
        loc_name = str(sub["location"].iloc[0])
        color = colors.get(iso, "#333333")

        pfig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["new_cases_smoothed_per_million"],
                mode="lines",
                name=f"{loc_name} (7d сглаж.)",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{loc_name}</b><br>Дата: %{{x|%Y-%m-%d}}<br>Случаев/млн: %{{y:.1f}}<extra></extra>",
            )
        )

        sub_waves = waves_df[waves_df["iso_code"] == iso]
        if not sub_waves.empty:
            pfig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sub_waves["peak_date"]),
                    y=sub_waves["peak_cases_pm_smoothed"],
                    mode="markers",
                    name=f"{loc_name} — пики волн",
                    marker=dict(color=color, size=9, symbol="diamond", line=dict(color="black", width=1)),
                    customdata=sub_waves[["wave_number", "prominence"]],
                    hovertemplate=(
                        f"<b>{loc_name} — Волна %{{customdata[0]}}</b><br>"
                        "Пик: %{x|%Y-%m-%d}<br>Случаев/млн: %{y:.1f}<br>"
                        "Prominence: %{customdata[1]:.1f}<extra></extra>"
                    ),
                )
            )

    pfig.update_layout(
        title="Интерактивное сравнение волн COVID-19 (клик по легенде включает/выключает страны)",
        xaxis_title="Дата",
        yaxis_title="Новые случаи на 1 млн (7-дневное сглаживание)",
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True)),
        height=600,
    )
    interactive_out = FIGURES_DIR / "fig5_interactive_waves.html"
    pfig.write_html(interactive_out)
    print(f"[5/5] Графики волн сохранены -> {static_out.name} и {interactive_out.name}")


def run_analytics_pipeline() -> None:
    """Полный запуск Шага 3: skewness/boxplot, временные ряды, корреляции, волны и графики."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_clean_countries()
        skew_df = analyze_skewness_and_plot_boxplots(df)
        weekly, monthly = analyze_time_series_and_decompose(df)
        corr_df = compute_correlations_and_heatmap(df)
        waves_df = detect_covid_waves(df)
        plot_waves_static_and_interactive(df, waves_df)

        summary = {
            "active_period": [ACTIVE_START, ACTIVE_END],
            "weekly_points": int(len(weekly)),
            "monthly_points": int(len(monthly)),
            "total_waves_detected": int(len(waves_df)),
            "countries_with_waves": int(waves_df["iso_code"].nunique()),
            "avg_waves_per_country": round(float(waves_df.groupby("iso_code").size().mean()), 2),
            "corr_vacc_vs_cfr": float(corr_df.loc["full_vacc_pct", "cfr_percent"]),
            "corr_gdp_vs_vacc": float(corr_df.loc["gdp_per_capita", "full_vacc_pct"]),
            "corr_median_age_vs_deaths_pm": float(corr_df.loc["median_age", "total_deaths_pm"]),
        }
        with open(ANALYTICS_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n--- Асимметрия (доказательство скошенности) ---")
        print(skew_df.to_string(index=False))

        print("\n--- Ключевые корреляции (уровень стран) ---")
        print(f"Вакцинация vs Летальность (CFR):   r = {summary['corr_vacc_vs_cfr']:.3f}")
        print(f"ВВП на душу vs Вакцинация:         r = {summary['corr_gdp_vs_vacc']:.3f}")
        print(f"Медианный возраст vs Смерти / млн: r = {summary['corr_median_age_vs_deaths_pm']:.3f}")

        print("\n--- Волны по Казахстану (KAZ) ---")
        kaz_waves = waves_df[waves_df["iso_code"] == "KAZ"]
        print(kaz_waves[["wave_number", "peak_date", "peak_cases_pm_smoothed", "prominence"]].to_string(index=False))

    except FileNotFoundError as exc:
        print(f"[ОШИБКА ФАЙЛА] {exc}")
    except Exception as exc:
        print(f"[НЕОЖИДАННАЯ ОШИБКА] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_analytics_pipeline()
