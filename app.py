import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from src.config import (
    CLEAN_AGGREGATES_PARQUET,
    CLEAN_COUNTRIES_PARQUET,
    DB_PATH,
    METRICS_DIR,
    OUTPUTS_DIR,
)

WAVES_PATH = METRICS_DIR / "waves_detected.csv"
FORECAST_METRICS_PATH = METRICS_DIR / "forecast_comparison.csv"
FORECAST_PREDS_PATH = METRICS_DIR / "forecast_predictions.csv"
SQL_OUTPUT_DIR = OUTPUTS_DIR / "sql"

# Единая цветовая палитра континентов для всех графиков (Вкладки 2 и 4)
CONTINENT_COLORS: Dict[str, str] = {
    "Asia": "#2563eb",
    "Europe": "#dc2626",
    "North America": "#16a34a",
    "South America": "#9333ea",
    "Africa": "#ea580c",
    "Oceania": "#0891b2",
}

OWID_PALETTE = [
    "#1d4ed8", "#c2410c", "#15803d", "#7e22ce",
    "#0f766e", "#be123c", "#a16207", "#334155",
]

# 6 метрик для режима Small Multiples (с зумом по клику)
FACET_METRICS: List[Tuple[str, str]] = [
    ("new_cases_smoothed_per_million", "Случаи / млн (7d)"),
    ("new_deaths_smoothed_per_million", "Смерти / млн (7d)"),
    ("people_fully_vaccinated_per_hundred", "Вакцинация (%)"),
    ("total_boosters_per_hundred", "Бустеры / 100 чел."),
    ("stringency_index", "Жёсткость мер (0–100)"),
    ("excess_mortality", "Избыт. смертность (%)"),
]

# Пресеты регионов для Вкладки 3 (Детекция волн)
WAVE_REGION_PRESETS: Dict[str, Dict] = {
    "CUSTOM": {"label": "Ручной выбор стран (мультивыбор ниже)", "isos": None},
    "GLOBAL": {"label": "Весь мир (Global — все 237 стран)", "isos": "ALL"},
    "CENTRAL_ASIA": {
        "label": "Центральная Азия (KAZ, UZB, KGZ, TJK, TKM)",
        "isos": ["KAZ", "UZB", "KGZ", "TJK", "TKM"],
    },
    "SE_ASIA": {
        "label": "Юго-Восточная Азия / Southeast Asia (10 стран)",
        "isos": ["SGP", "MYS", "THA", "IDN", "VNM", "PHL", "MMR", "KHM", "LAO", "BRN"],
    },
    "AMERICAS": {
        "label": "Америка (North + South America)",
        "continents": ["North America", "South America"],
    },
    "CONT::Asia": {"label": "Континент: Азия (Asia)", "continents": ["Asia"]},
    "CONT::Europe": {"label": "Континент: Европа (Europe)", "continents": ["Europe"]},
    "CONT::North America": {"label": "Континент: Северная Америка (North America)", "continents": ["North America"]},
    "CONT::South America": {"label": "Континент: Южная Америка (South America)", "continents": ["South America"]},
    "CONT::Africa": {"label": "Континент: Африка (Africa)", "continents": ["Africa"]},
    "CONT::Oceania": {"label": "Континент: Океания (Oceania)", "continents": ["Oceania"]},
}

# 14 метрик для OWID Explorer
EXPLORER_METRICS: Dict[str, Dict] = {
    "excess_mortality": {
        "label": "Excess mortality: P-score (% vs baseline)",
        "subtitle": "Процентная разница между зафиксированными и ожидаемыми смертями от всех причин (аналог OWID P-score).",
        "unit": "%",
        "colorscale": "RdYlBu_r",
        "diverging_zero": True,
        "agg": "last",
        "ffill_months": 6,
    },
    "excess_mortality_cumulative": {
        "label": "Excess mortality: Cumulative (% vs projection)",
        "subtitle": "Накопленная избыточная смертность в процентах относительно допандемийного прогноза.",
        "unit": "%",
        "colorscale": "RdYlBu_r",
        "diverging_zero": True,
        "agg": "last",
        "ffill_months": 6,
    },
    "excess_mortality_cumulative_per_million": {
        "label": "Excess mortality: Cumulative per million",
        "subtitle": "Накопленная избыточная смертность на 1 млн населения (идеально для сравнения 2 дат на карте).",
        "unit": "смертей / млн",
        "colorscale": "YlOrRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "excess_mortality_cumulative_absolute": {
        "label": "Excess mortality: Cumulative absolute deaths",
        "subtitle": "Абсолютное накопленное число избыточных смертей от всех причин с начала 2020 года.",
        "unit": "смертей",
        "colorscale": "YlOrRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "new_cases_smoothed_per_million": {
        "label": "Daily confirmed cases per million (7-day avg)",
        "subtitle": "Новые подтверждённые случаи COVID-19 на 1 млн населения (7-дневное сглаживание).",
        "unit": "случаев / млн",
        "colorscale": "Blues",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "new_deaths_smoothed_per_million": {
        "label": "Daily confirmed deaths per million (7-day avg)",
        "subtitle": "Новые подтверждённые смерти от COVID-19 на 1 млн населения (7-дневное сглаживание).",
        "unit": "смертей / млн",
        "colorscale": "Reds",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "total_cases_per_million": {
        "label": "Cumulative confirmed cases per million",
        "subtitle": "Суммарное число подтверждённых случаев на 1 млн жителей.",
        "unit": "случаев / млн",
        "colorscale": "Blues",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "total_deaths_per_million": {
        "label": "Cumulative confirmed deaths per million",
        "subtitle": "Суммарное число подтверждённых смертей от COVID-19 на 1 млн жителей.",
        "unit": "смертей / млн",
        "colorscale": "OrRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "people_fully_vaccinated_per_hundred": {
        "label": "Share of people fully vaccinated (%)",
        "subtitle": "Доля населения, завершившего первичный курс вакцинации против COVID-19.",
        "unit": "%",
        "colorscale": "Greens",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 12,
    },
    "total_boosters": {
        "label": "COVID-19 vaccine boosters administered (total)",
        "subtitle": "Общее число введённых бустерных доз вакцины.",
        "unit": "доз",
        "colorscale": "PuRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 12,
    },
    "total_boosters_per_hundred": {
        "label": "COVID-19 vaccine boosters per 100 people",
        "subtitle": "Число введённых бустерных доз на 100 человек населения.",
        "unit": "доз / 100 чел.",
        "colorscale": "PuRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 12,
    },
    "icu_patients_per_million": {
        "label": "COVID-19 patients in ICU per million",
        "subtitle": "Число пациентов с COVID-19 в отделениях интенсивной терапии на 1 млн человек.",
        "unit": "пациентов / млн",
        "colorscale": "Purples",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "hosp_patients": {
        "label": "COVID-19 patients in hospital",
        "subtitle": "Текущее число госпитализированных пациентов с COVID-19.",
        "unit": "пациентов",
        "colorscale": "YlOrRd",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "stringency_index": {
        "label": "COVID-19 Stringency Index (0–100)",
        "subtitle": "Индекс жёсткости противоэпидемических мер (100 = максимум).",
        "unit": "баллов",
        "colorscale": "YlGn",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 2,
    },
}

MAP_SCOPE_OPTIONS = [
    {"label": "World", "value": "world"},
    {"label": "Asia", "value": "asia"},
    {"label": "Europe", "value": "europe"},
    {"label": "Africa", "value": "africa"},
    {"label": "North America", "value": "north america"},
    {"label": "South America", "value": "south america"},
]

SCOPE_TO_CONTINENT = {
    "asia": "Asia",
    "europe": "Europe",
    "africa": "Africa",
    "north america": "North America",
    "south america": "South America",
}

CATEGORY_META: Dict[str, Dict[str, str]] = {
    "cases_pm": {
        "label": "Заболеваемость (случаи / млн)",
        "table": "epidemiology",
        "per_million_col": "new_cases_per_million",
        "agg": "mean",
        "unit": "случаев / млн в день",
    },
    "deaths_pm": {
        "label": "Смертность (смерти / млн)",
        "table": "epidemiology",
        "per_million_col": "new_deaths_per_million",
        "agg": "mean",
        "unit": "смертей / млн в день",
    },
    "vacc_pct": {
        "label": "Вакцинация (полный курс, %)",
        "table": "vaccinations",
        "per_million_col": "people_fully_vaccinated_per_hundred",
        "agg": "max",
        "unit": "% населения",
    },
    "stringency": {
        "label": "Жёсткость мер (Stringency Index)",
        "table": "hospital_policy",
        "per_million_col": "stringency_index",
        "agg": "mean",
        "unit": "баллов (0–100)",
    },
}


def query_sqlite(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except sqlite3.Error as exc:
        print(f"[Dashboard SQL Error] {exc}")
        return pd.DataFrame()


def format_compact_val(val: float, metric_key: str) -> str:
    if pd.isna(val):
        return "No data"
    meta = EXPLORER_METRICS[metric_key]
    if meta["unit"] == "%":
        return f"{val:.1f}%"
    av = abs(val)
    if av >= 1e9:
        return f"{val / 1e9:.2f}B"
    if av >= 1e6:
        return f"{val / 1e6:.2f}M"
    if av >= 1e4:
        return f"{val:,.0f}"
    if av >= 100:
        return f"{val:,.1f}"
    return f"{val:.2f}"


def empty_figure(message: str, height: int = 440) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=15, color="#475569"),
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
    )
    return fig


def load_explorer_store() -> Dict:
    """Готовит быстрые недельные и месячные таблицы для OWID Explorer из Parquet."""
    metric_cols = list(EXPLORER_METRICS.keys())
    base_cols = ["iso_code", "continent", "location", "date", "population", "gdp_per_capita"]

    frames: List[pd.DataFrame] = []
    for path, is_agg in [(CLEAN_COUNTRIES_PARQUET, False), (CLEAN_AGGREGATES_PARQUET, True)]:
        if not path.exists():
            continue
        raw = pd.read_parquet(path)
        avail = [c for c in base_cols + metric_cols if c in raw.columns]
        part = raw[avail].copy()
        for c in ["iso_code", "continent", "location"]:
            if c in part.columns:
                part[c] = part[c].astype(object).where(part[c].notna(), "").astype(str)
        part["is_aggregate"] = is_agg
        if is_agg:
            part["continent"] = "Regions & Income Groups"
            part["entity_id"] = "AGG::" + part["location"]
        else:
            part["entity_id"] = part["iso_code"]
        frames.append(part)

    if not frames:
        return {"entities": pd.DataFrame(), "weekly": pd.DataFrame(), "monthly": pd.DataFrame(), "months": []}

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-08-31")].copy()

    entities = (
        df.groupby("entity_id", as_index=False)
        .agg(
            iso_code=("iso_code", "first"),
            location=("location", "first"),
            continent=("continent", "first"),
            is_aggregate=("is_aggregate", "first"),
            population=("population", "max"),
            gdp_per_capita=("gdp_per_capita", "max"),
        )
        .sort_values("location")
        .reset_index(drop=True)
    )

    df["week_date"] = df["date"].dt.to_period("W").dt.end_time.dt.normalize()
    agg_dict = {
        col: ("mean" if EXPLORER_METRICS[col]["agg"] == "mean" else "last")
        for col in metric_cols
        if col in df.columns
    }

    weekly = (
        df.sort_values("date")
        .groupby(["entity_id", "week_date"], as_index=False)
        .agg(agg_dict)
    )

    df["month"] = df["date"].dt.strftime("%Y-%m")
    monthly = (
        df.sort_values("date")
        .groupby(["entity_id", "month"], as_index=False)
        .agg(agg_dict)
    )

    months = sorted(monthly["month"].unique().tolist())

    full_idx = pd.MultiIndex.from_product([entities["entity_id"].tolist(), months], names=["entity_id", "month"])
    monthly_grid = monthly.set_index(["entity_id", "month"]).reindex(full_idx)
    monthly_filled = pd.DataFrame(index=full_idx)
    for col in metric_cols:
        if col in monthly_grid.columns:
            lim = EXPLORER_METRICS[col]["ffill_months"]
            monthly_filled[col] = monthly_grid[col].groupby(level=0).ffill(limit=lim)

    monthly_filled = monthly_filled.reset_index().merge(
        entities[["entity_id", "iso_code", "location", "continent", "is_aggregate", "population", "gdp_per_capita"]],
        on="entity_id",
        how="left",
    )

    return {"entities": entities, "weekly": weekly, "monthly": monthly_filled, "months": months}


def load_static_artifacts() -> Dict[str, pd.DataFrame]:
    artifacts: Dict[str, pd.DataFrame] = {}
    artifacts["locations"] = query_sqlite(
        "SELECT iso_code, continent, location, population, gdp_per_capita FROM locations ORDER BY location"
    )
    artifacts["waves"] = pd.read_csv(WAVES_PATH) if WAVES_PATH.exists() else pd.DataFrame()
    artifacts["forecast_metrics"] = (
        pd.read_csv(FORECAST_METRICS_PATH) if FORECAST_METRICS_PATH.exists() else pd.DataFrame()
    )
    artifacts["forecast_preds"] = (
        pd.read_csv(FORECAST_PREDS_PATH) if FORECAST_PREDS_PATH.exists() else pd.DataFrame()
    )
    return artifacts


def list_sql_csv_files() -> List[str]:
    if not SQL_OUTPUT_DIR.exists():
        return []
    return sorted([p.stem for p in SQL_OUTPUT_DIR.glob("*.csv")])


def build_year_bounds(year_filter: str) -> Tuple[str, str]:
    if year_filter == "ACTIVE":
        return "2020-01-01", "2023-12-31"
    if year_filter == "ALL":
        return "2020-01-01", "2026-12-31"
    return f"{year_filter}-01-01", f"{year_filter}-12-31"


print("[Dashboard] Загрузка данных и подготовка OWID Explorer...")
ARTIFACTS = load_static_artifacts()
EXPLORER = load_explorer_store()
MONTHS: List[str] = EXPLORER["months"]
DEFAULT_START_IDX = MONTHS.index("2021-09") if "2021-09" in MONTHS else 0
DEFAULT_END_IDX = MONTHS.index("2023-12") if "2023-12" in MONTHS else max(len(MONTHS) - 1, 0)

# Без дублирования года (только январь каждого года)
SLIDER_MARKS = {
    i: {"label": m[:4], "style": {"fontSize": "11px", "color": "#475569"}}
    for i, m in enumerate(MONTHS)
    if m.endswith("-01")
}

CONTINENTS = (
    sorted(ARTIFACTS["locations"]["continent"].dropna().unique().tolist())
    if not ARTIFACTS["locations"].empty
    else []
)
WAVE_COUNTRY_OPTIONS = (
    [
        {"label": f"{row['location']} ({row['iso_code']})", "value": row["iso_code"]}
        for _, row in ARTIFACTS["locations"].iterrows()
    ]
    if not ARTIFACTS["locations"].empty
    else [{"label": "Kazakhstan (KAZ)", "value": "KAZ"}]
)
FORECAST_SERIES_OPTIONS = (
    [
        {"label": f"{row['series_name']} ({row['series_code']})", "value": row["series_code"]}
        for _, row in ARTIFACTS["forecast_metrics"][["series_code", "series_name"]].drop_duplicates().iterrows()
    ]
    if not ARTIFACTS["forecast_metrics"].empty
    else [{"label": "Global", "value": "GLOBAL"}]
)
FORECAST_MODEL_OPTIONS = [{"label": "Авто: лучшая модель ряда (min MAE)", "value": "AUTO_BEST"}] + [
    {"label": m, "value": m}
    for m in [
        "Linear Regression (AR lags)",
        "Naive (baseline)",
        "Moving Average 7d",
        "Seasonal Naive 7d (baseline)",
    ]
]
SQL_QUERY_OPTIONS = [{"label": name, "value": name} for name in list_sql_csv_files()]

app = Dash(__name__, title="COVID-19 Global Dynamics — OWID Explorer & Analytics")

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #e2e8f0",
    "borderRadius": "10px",
    "padding": "14px 18px",
    "flex": "1",
    "minWidth": "180px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}

PANEL_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #e2e8f0",
    "borderRadius": "10px",
    "padding": "16px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
}

FACET_PILL_STYLE = {
    "padding": "4px 10px",
    "border": "1px solid #cbd5e1",
    "borderRadius": "16px",
    "backgroundColor": "#f8fafc",
    "cursor": "pointer",
    "fontSize": "12px",
    "fontWeight": "500",
    "color": "#1e293b",
}

app.layout = html.Div(
    style={
        "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "backgroundColor": "#f8fafc",
        "minHeight": "100vh",
        "padding": "18px 26px",
        "color": "#0f172a",
    },
    children=[
        dcc.Download(id="explorer-download"),
        dcc.Interval(id="explorer-play-interval", interval=850, n_intervals=0, disabled=True),
        dcc.Store(id="explorer-sort-dir", data="desc"),
        dcc.Store(id="explorer-facet-zoom", data=None),
        # Шапка
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "marginBottom": "14px",
            },
            children=[
                html.Div(
                    children=[
                        html.H1(
                            "COVID-19 Global Dynamics — Interactive Explorer & Analytics",
                            style={"margin": "0 0 4px 0", "fontSize": "24px", "color": "#0f172a"},
                        ),
                        html.P(
                            "Источник: Our World in Data (compact.csv) | Хранилище: Parquet + SQLite (covid.db) | Волны, избыточная смертность, вакцинация и прогноз",
                            style={"margin": "0", "color": "#475569", "fontSize": "13px"},
                        ),
                    ]
                ),
                html.Div(
                    "OWID + ML",
                    style={
                        "backgroundColor": "#0b2545",
                        "color": "#ffffff",
                        "padding": "6px 12px",
                        "borderRadius": "4px",
                        "fontWeight": "700",
                        "fontSize": "12px",
                        "borderBottom": "3px solid #dc2626",
                    },
                ),
            ],
        ),
        # Вкладки
        dcc.Tabs(
            id="main-tabs",
            value="tab-explorer",
            children=[
                # ==================== ВКЛАДКА 1: OWID EXPLORER ====================
                dcc.Tab(
                    label="1. OWID Explorer (Карты, Сравнение, Линии)",
                    value="tab-explorer",
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "minmax(0, 1fr) 335px",
                                "gap": "16px",
                                "marginTop": "14px",
                                "alignItems": "start",
                            },
                            children=[
                                # Левая основная панель
                                html.Div(
                                    style=PANEL_STYLE,
                                    children=[
                                        html.Div(
                                            style={"marginBottom": "10px"},
                                            children=[
                                                html.Div(
                                                    id="explorer-title",
                                                    style={
                                                        "fontSize": "20px",
                                                        "fontWeight": "700",
                                                        "color": "#0f172a",
                                                        "fontFamily": "Georgia, serif",
                                                    },
                                                ),
                                                html.Div(
                                                    id="explorer-subtitle",
                                                    style={"fontSize": "13px", "color": "#475569", "marginTop": "3px"},
                                                ),
                                            ],
                                        ),
                                        # Переключатель режимов + фильтр карты + Reset view + регион карты
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "alignItems": "center",
                                                "flexWrap": "wrap",
                                                "gap": "10px",
                                                "marginBottom": "12px",
                                                "paddingBottom": "10px",
                                                "borderBottom": "1px solid #e2e8f0",
                                            },
                                            children=[
                                                dcc.RadioItems(
                                                    id="explorer-view-mode",
                                                    options=[
                                                        {"label": "Map", "value": "map"},
                                                        {"label": "Compare 2 Dates", "value": "compare_map"},
                                                        {"label": "Line", "value": "line"},
                                                        {"label": "Year Overlay", "value": "year_overlay"},
                                                        {"label": "Small Multiples", "value": "facets"},
                                                        {"label": "Bar", "value": "bar"},
                                                        {"label": "Table", "value": "table"},
                                                    ],
                                                    value="compare_map",
                                                    inline=True,
                                                    labelStyle={
                                                        "display": "inline-block",
                                                        "padding": "5px 10px",
                                                        "marginRight": "6px",
                                                        "border": "1px solid #cbd5e1",
                                                        "borderRadius": "6px",
                                                        "cursor": "pointer",
                                                        "fontSize": "12px",
                                                        "fontWeight": "600",
                                                        "backgroundColor": "#f8fafc",
                                                    },
                                                    inputStyle={"marginRight": "5px"},
                                                ),
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
                                                    children=[
                                                        dcc.Checklist(
                                                            id="explorer-map-only-selected",
                                                            options=[{"label": "Только выбранные на карте", "value": "only"}],
                                                            value=["only"],
                                                            style={"fontSize": "12px", "color": "#334155"},
                                                            inputStyle={"marginRight": "5px"},
                                                        ),
                                                        html.Button(
                                                            "Reset view",
                                                            id="explorer-reset-view",
                                                            n_clicks=0,
                                                            style={
                                                                "padding": "5px 10px",
                                                                "border": "1px solid #cbd5e1",
                                                                "borderRadius": "6px",
                                                                "backgroundColor": "#ffffff",
                                                                "cursor": "pointer",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            style={"width": "148px"},
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="explorer-map-scope",
                                                                    options=MAP_SCOPE_OPTIONS,
                                                                    value="world",
                                                                    clearable=False,
                                                                    style={"fontSize": "12px"},
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        # Панель зума/раззума для Small Multiples (всегда в DOM, скрывается вне режима facets)
                                        html.Div(
                                            id="explorer-facet-bar",
                                            style={"display": "none", "marginBottom": "10px"},
                                            children=[
                                                html.Div(
                                                    id="facet-grid-controls",
                                                    style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "6px"},
                                                    children=[
                                                        html.Span(
                                                            "🔍 Увеличить график (кликните по линии или кнопке):",
                                                            style={"fontSize": "12px", "color": "#475569", "fontWeight": "600", "marginRight": "4px"},
                                                        ),
                                                        *[
                                                            html.Button(
                                                                title,
                                                                id=f"facet-btn-{idx}",
                                                                n_clicks=0,
                                                                style=FACET_PILL_STYLE,
                                                            )
                                                            for idx, (_, title) in enumerate(FACET_METRICS)
                                                        ],
                                                    ],
                                                ),
                                                html.Div(
                                                    id="facet-zoom-controls",
                                                    style={"display": "none", "alignItems": "center", "gap": "12px"},
                                                    children=[
                                                        html.Button(
                                                            "← Вернуться ко всем 6 графикам",
                                                            id="facet-back-btn",
                                                            n_clicks=0,
                                                            style={
                                                                "padding": "6px 14px",
                                                                "border": "1px solid #1d4ed8",
                                                                "borderRadius": "6px",
                                                                "backgroundColor": "#eff6ff",
                                                                "color": "#1d4ed8",
                                                                "cursor": "pointer",
                                                                "fontSize": "12px",
                                                                "fontWeight": "700",
                                                            },
                                                        ),
                                                        html.Span(
                                                            id="facet-zoom-label",
                                                            style={"fontSize": "13px", "fontWeight": "600", "color": "#0f172a"},
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        # График / Таблица
                                        html.Div(
                                            id="explorer-graph-wrapper",
                                            children=[dcc.Graph(id="explorer-graph", config={"displaylogo": False})],
                                        ),
                                        html.Div(id="explorer-table-wrapper", style={"display": "none"}),
                                        # Нижний таймлайн с Play
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "gap": "12px",
                                                "marginTop": "12px",
                                                "paddingTop": "10px",
                                                "borderTop": "1px solid #e2e8f0",
                                            },
                                            children=[
                                                html.Button(
                                                    "▶ Play",
                                                    id="explorer-play-btn",
                                                    n_clicks=0,
                                                    style={
                                                        "padding": "6px 12px",
                                                        "border": "1px solid #cbd5e1",
                                                        "borderRadius": "6px",
                                                        "backgroundColor": "#f1f5f9",
                                                        "cursor": "pointer",
                                                        "fontWeight": "600",
                                                        "fontSize": "12px",
                                                        "minWidth": "74px",
                                                    },
                                                ),
                                                html.Span(
                                                    id="explorer-start-label",
                                                    style={"fontSize": "12px", "fontWeight": "600", "color": "#334155", "minWidth": "62px"},
                                                ),
                                                html.Div(
                                                    style={"flex": "1", "padding": "0 6px"},
                                                    children=[
                                                        dcc.RangeSlider(
                                                            id="explorer-slider",
                                                            min=0,
                                                            max=max(len(MONTHS) - 1, 1),
                                                            step=1,
                                                            value=[DEFAULT_START_IDX, DEFAULT_END_IDX],
                                                            marks=SLIDER_MARKS,
                                                            allowCross=False,
                                                        )
                                                    ],
                                                ),
                                                html.Span(
                                                    id="explorer-end-label",
                                                    style={"fontSize": "12px", "fontWeight": "600", "color": "#334155", "minWidth": "62px"},
                                                ),
                                            ],
                                        ),
                                        # Подвал + Download CSV
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "alignItems": "center",
                                                "marginTop": "10px",
                                                "fontSize": "11px",
                                                "color": "#64748b",
                                            },
                                            children=[
                                                html.Span(
                                                    "Data source: Our World in Data (compact.csv). Note: where exact month is unavailable, closest value within lookback window is shown."
                                                ),
                                                html.Button(
                                                    "⬇ Download CSV",
                                                    id="explorer-download-btn",
                                                    n_clicks=0,
                                                    style={
                                                        "padding": "5px 10px",
                                                        "border": "1px solid #cbd5e1",
                                                        "borderRadius": "6px",
                                                        "backgroundColor": "#ffffff",
                                                        "cursor": "pointer",
                                                        "fontSize": "12px",
                                                        "fontWeight": "600",
                                                        "color": "#0f172a",
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Правая панель OWID
                                html.Div(
                                    style=PANEL_STYLE,
                                    children=[
                                        html.Div(
                                            "METRIC / INDICATOR",
                                            style={"fontSize": "11px", "fontWeight": "700", "color": "#64748b", "letterSpacing": "0.05em", "marginBottom": "6px"},
                                        ),
                                        dcc.Dropdown(
                                            id="explorer-metric",
                                            options=[{"label": v["label"], "value": k} for k, v in EXPLORER_METRICS.items()],
                                            value="excess_mortality_cumulative_per_million",
                                            clearable=False,
                                            style={"fontSize": "12px", "marginBottom": "14px"},
                                        ),
                                        html.Div(
                                            "ADD/REMOVE COUNTRIES AND REGIONS",
                                            style={"fontSize": "11px", "fontWeight": "700", "color": "#64748b", "letterSpacing": "0.05em", "marginBottom": "6px"},
                                        ),
                                        dcc.Input(
                                            id="explorer-search",
                                            type="text",
                                            placeholder="🔍 Search for a country or region...",
                                            debounce=False,
                                            value="",
                                            style={
                                                "width": "100%",
                                                "boxSizing": "border-box",
                                                "padding": "7px 10px",
                                                "border": "1px solid #cbd5e1",
                                                "borderRadius": "6px",
                                                "fontSize": "12px",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            style={"display": "flex", "gap": "6px", "marginBottom": "8px"},
                                            children=[
                                                html.Div(
                                                    style={"flex": "1"},
                                                    children=[
                                                        dcc.Dropdown(
                                                            id="explorer-sort-by",
                                                            options=[
                                                                {"label": "Sort by: Current metric", "value": "metric"},
                                                                {"label": "Sort by: Name", "value": "location"},
                                                                {"label": "Sort by: Population", "value": "population"},
                                                                {"label": "Sort by: GDP per capita", "value": "gdp_per_capita"},
                                                            ],
                                                            value="metric",
                                                            clearable=False,
                                                            style={"fontSize": "12px"},
                                                        )
                                                    ],
                                                ),
                                                html.Button(
                                                    "↓",
                                                    id="explorer-sort-toggle",
                                                    n_clicks=0,
                                                    title="Toggle sort direction",
                                                    style={
                                                        "width": "34px",
                                                        "border": "1px solid #cbd5e1",
                                                        "borderRadius": "6px",
                                                        "backgroundColor": "#f8fafc",
                                                        "cursor": "pointer",
                                                        "fontWeight": "700",
                                                    },
                                                ),
                                            ],
                                        ),
                                        dcc.Dropdown(
                                            id="explorer-type-filter",
                                            options=[
                                                {"label": "Filter: All (Countries + Regions)", "value": "ALL"},
                                                {"label": "Filter: Countries only", "value": "COUNTRIES"},
                                                {"label": "Filter: Continents & Income groups", "value": "AGGREGATES"},
                                            ]
                                            + [{"label": f"Continent: {c}", "value": f"CONT::{c}"} for c in CONTINENTS],
                                            value="COUNTRIES",
                                            clearable=False,
                                            style={"fontSize": "12px", "marginBottom": "10px"},
                                        ),
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "justifyContent": "space-between",
                                                "alignItems": "center",
                                                "marginBottom": "6px",
                                                "fontSize": "12px",
                                            },
                                            children=[
                                                html.Span(id="explorer-selection-count", style={"fontWeight": "600", "color": "#334155"}),
                                                html.Button(
                                                    "Clear",
                                                    id="explorer-clear-btn",
                                                    n_clicks=0,
                                                    style={
                                                        "border": "none",
                                                        "background": "transparent",
                                                        "color": "#1d4ed8",
                                                        "textDecoration": "underline",
                                                        "cursor": "pointer",
                                                        "fontSize": "12px",
                                                    },
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={
                                                "maxHeight": "430px",
                                                "overflowY": "auto",
                                                "border": "1px solid #e2e8f0",
                                                "borderRadius": "6px",
                                                "padding": "6px 8px",
                                                "backgroundColor": "#f8fafc",
                                            },
                                            children=[
                                                dcc.Checklist(
                                                    id="explorer-countries",
                                                    options=[],
                                                    value=["KAZ", "USA", "GBR", "RUS", "DEU"],
                                                    labelStyle={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                        "width": "100%",
                                                        "padding": "4px 2px",
                                                        "borderBottom": "1px solid #eef2f6",
                                                        "fontSize": "12px",
                                                        "cursor": "pointer",
                                                    },
                                                    inputStyle={"marginRight": "8px", "flexShrink": "0"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                # ==================== ВКЛАДКА 2: ОБЗОР SQLITE ====================
                dcc.Tab(
                    label="2. Обзор и динамика (SQLite)",
                    value="tab-overview",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "16px",
                                "flexWrap": "wrap",
                                "backgroundColor": "#ffffff",
                                "padding": "14px 18px",
                                "borderRadius": "10px",
                                "border": "1px solid #e2e8f0",
                                "marginTop": "14px",
                            },
                            children=[
                                html.Div(
                                    style={"flex": "1", "minWidth": "200px"},
                                    children=[
                                        html.Label("Год (период):", style={"fontWeight": "600", "fontSize": "13px"}),
                                        dcc.Dropdown(
                                            id="filter-year",
                                            options=[
                                                {"label": "Активная фаза (2020–2023)", "value": "ACTIVE"},
                                                {"label": "2020", "value": "2020"},
                                                {"label": "2021", "value": "2021"},
                                                {"label": "2022", "value": "2022"},
                                                {"label": "2023", "value": "2023"},
                                                {"label": "Весь датасет (2020–2026)", "value": "ALL"},
                                            ],
                                            value="ACTIVE",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": "1", "minWidth": "200px"},
                                    children=[
                                        html.Label("Регион (континент):", style={"fontWeight": "600", "fontSize": "13px"}),
                                        dcc.Dropdown(
                                            id="filter-region",
                                            options=[{"label": "Все континенты (мир)", "value": "ALL"}]
                                            + [{"label": c, "value": c} for c in CONTINENTS],
                                            value="ALL",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"flex": "1.3", "minWidth": "240px"},
                                    children=[
                                        html.Label("Категория (метрика):", style={"fontWeight": "600", "fontSize": "13px"}),
                                        dcc.Dropdown(
                                            id="filter-category",
                                            options=[{"label": v["label"], "value": k} for k, v in CATEGORY_META.items()],
                                            value="cases_pm",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="kpi-row",
                            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "margin": "14px 0"},
                        ),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1.35fr 1fr", "gap": "16px"},
                            children=[
                                html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="overview-timeseries")]),
                                html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="overview-top-countries")]),
                            ],
                        ),
                    ],
                ),
                # ==================== ВКЛАДКА 3: ДЕТЕКЦИЯ ВОЛН (СТРАНЫ + РЕГИОНЫ) ====================
                dcc.Tab(
                    label="3. Детекция волн",
                    value="tab-waves",
                    children=[
                        html.Div(
                            style={**PANEL_STYLE, "marginTop": "14px"},
                            children=[
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1fr 1.5fr auto",
                                        "gap": "14px",
                                        "alignItems": "end",
                                        "marginBottom": "12px",
                                    },
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Label("Пресет региона / континента:", style={"fontWeight": "600", "fontSize": "13px"}),
                                                dcc.Dropdown(
                                                    id="wave-preset",
                                                    options=[{"label": v["label"], "value": k} for k, v in WAVE_REGION_PRESETS.items()],
                                                    value="CUSTOM",
                                                    clearable=False,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Или выберите одну / несколько стран вручную:", style={"fontWeight": "600", "fontSize": "13px"}),
                                                dcc.Dropdown(
                                                    id="wave-countries",
                                                    options=WAVE_COUNTRY_OPTIONS,
                                                    value=["KAZ"],
                                                    multi=True,
                                                    placeholder="Выберите страны (например KAZ, UZB, KGZ)...",
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Label("Режим отображения:", style={"fontWeight": "600", "fontSize": "13px", "display": "block", "marginBottom": "4px"}),
                                                dcc.RadioItems(
                                                    id="wave-agg-mode",
                                                    options=[
                                                        {"label": "Суммарная волна", "value": "aggregate"},
                                                        {"label": "По странам отдельно", "value": "compare"},
                                                    ],
                                                    value="aggregate",
                                                    inline=True,
                                                    labelStyle={
                                                        "display": "inline-block",
                                                        "padding": "6px 10px",
                                                        "marginRight": "6px",
                                                        "border": "1px solid #cbd5e1",
                                                        "borderRadius": "6px",
                                                        "fontSize": "12px",
                                                        "fontWeight": "600",
                                                        "backgroundColor": "#f8fafc",
                                                        "cursor": "pointer",
                                                    },
                                                    inputStyle={"marginRight": "5px"},
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                html.Div(id="wave-scope-badge", style={"fontSize": "12px", "color": "#475569", "marginBottom": "8px"}),
                                dcc.Graph(id="waves-graph"),
                                html.H4("Таблица обнаруженных пиков волн", style={"marginTop": "14px", "marginBottom": "8px"}),
                                html.Div(id="waves-table-container"),
                            ],
                        ),
                    ],
                ),
                # ==================== ВКЛАДКА 4: ВАКЦИНАЦИЯ И ПРОГНОЗ ====================
                dcc.Tab(
                    label="4. Вакцинация и прогноз",
                    value="tab-vacc-forecast",
                    children=[
                        html.Div(
                            style={**PANEL_STYLE, "marginTop": "14px"},
                            children=[dcc.Graph(id="vacc-mortality-scatter")],
                        ),
                        html.Div(
                            style={**PANEL_STYLE, "marginTop": "14px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "12px"},
                                    children=[
                                        html.Div(
                                            style={"minWidth": "300px", "flex": "1", "maxWidth": "380px"},
                                            children=[
                                                html.Label("Ряд для сравнения моделей прогноза (тест: июль–окт 2022):", style={"fontWeight": "600", "fontSize": "13px"}),
                                                dcc.Dropdown(
                                                    id="forecast-series",
                                                    options=FORECAST_SERIES_OPTIONS,
                                                    value="KAZ",
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={"minWidth": "280px", "flex": "1", "maxWidth": "360px"},
                                            children=[
                                                html.Label("Подсветить модель (остальные приглушены):", style={"fontWeight": "600", "fontSize": "13px"}),
                                                dcc.Dropdown(
                                                    id="forecast-focus-model",
                                                    options=FORECAST_MODEL_OPTIONS,
                                                    value="AUTO_BEST",
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                dcc.Graph(id="forecast-graph"),
                                html.H4("Сравнение качества (MAE, RMSE, nMAE %), времени и памяти", style={"marginTop": "14px", "marginBottom": "8px"}),
                                html.Div(id="forecast-table-container"),
                            ],
                        ),
                    ],
                ),
                # ==================== ВКЛАДКА 5: SQL-ВИТРИНА ====================
                dcc.Tab(
                    label="5. SQL-витрина (covid.db)",
                    value="tab-sql",
                    children=[
                        html.Div(
                            style={**PANEL_STYLE, "marginTop": "14px"},
                            children=[
                                html.Div(
                                    style={"maxWidth": "420px", "marginBottom": "12px"},
                                    children=[
                                        html.Label("Выберите результат SQL-запроса (9 аналитических запросов):", style={"fontWeight": "600", "fontSize": "13px"}),
                                        dcc.Dropdown(
                                            id="sql-selector",
                                            options=SQL_QUERY_OPTIONS,
                                            value=SQL_QUERY_OPTIONS[0]["value"] if SQL_QUERY_OPTIONS else None,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(id="sql-info", style={"marginBottom": "10px", "color": "#475569", "fontSize": "13px"}),
                                html.Div(id="sql-table-container"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ==================== КОЛБЭКИ OWID EXPLORER ====================

@app.callback(
    Output("explorer-sort-dir", "data"),
    Output("explorer-sort-toggle", "children"),
    Input("explorer-sort-toggle", "n_clicks"),
    State("explorer-sort-dir", "data"),
    prevent_initial_call=True,
)
def toggle_sort_direction(_n, current_dir: str):
    new_dir = "asc" if current_dir == "desc" else "desc"
    return new_dir, ("↑" if new_dir == "asc" else "↓")


@app.callback(
    Output("explorer-map-scope", "value"),
    Input("explorer-reset-view", "n_clicks"),
    prevent_initial_call=True,
)
def reset_map_scope(_n):
    return "world"


@app.callback(
    Output("explorer-play-interval", "disabled"),
    Output("explorer-play-btn", "children"),
    Input("explorer-play-btn", "n_clicks"),
    State("explorer-play-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_play(_n, is_disabled: bool):
    now_disabled = not is_disabled
    return now_disabled, ("▶ Play" if now_disabled else "⏸ Pause")


@app.callback(
    Output("explorer-slider", "value"),
    Input("explorer-play-interval", "n_intervals"),
    State("explorer-slider", "value"),
    prevent_initial_call=True,
)
def advance_slider(_n, slider_val: List[int]):
    if not MONTHS:
        return slider_val
    start_i, end_i = slider_val[0], slider_val[1]
    next_end = end_i + 1
    if next_end >= len(MONTHS):
        next_end = min(start_i + 1, len(MONTHS) - 1)
    return [start_i, next_end]


@app.callback(
    Output("explorer-facet-zoom", "data"),
    Input("explorer-graph", "clickData"),
    Input("facet-back-btn", "n_clicks"),
    Input("explorer-view-mode", "value"),
    *[Input(f"facet-btn-{i}", "n_clicks") for i in range(len(FACET_METRICS))],
    State("explorer-facet-zoom", "data"),
    prevent_initial_call=True,
)
def handle_facet_zoom(click_data, _back, view_mode, *args):
    current_zoom = args[-1]
    trig = ctx.triggered_id

    if trig in ("explorer-view-mode", "facet-back-btn"):
        return None

    if isinstance(trig, str) and trig.startswith("facet-btn-"):
        idx = int(trig.split("-")[-1])
        return FACET_METRICS[idx][0]

    if trig == "explorer-graph" and view_mode == "facets" and current_zoom is None and click_data:
        pts = click_data.get("points") or []
        if pts and "customdata" in pts[0]:
            cd = pts[0]["customdata"]
            metric_key = cd[0] if isinstance(cd, (list, tuple)) else cd
            if metric_key in EXPLORER_METRICS:
                return metric_key

    return current_zoom


@app.callback(
    Output("explorer-countries", "options"),
    Output("explorer-countries", "value"),
    Output("explorer-selection-count", "children"),
    Input("explorer-metric", "value"),
    Input("explorer-slider", "value"),
    Input("explorer-search", "value"),
    Input("explorer-sort-by", "value"),
    Input("explorer-sort-dir", "data"),
    Input("explorer-type-filter", "value"),
    Input("explorer-map-scope", "value"),
    Input("explorer-clear-btn", "n_clicks"),
    Input("explorer-countries", "value"),
)
def update_country_sidebar(
    metric: str,
    slider_val: List[int],
    search_text: str,
    sort_by: str,
    sort_dir: str,
    type_filter: str,
    map_scope: str,
    _clear_clicks: int,
    selected: Optional[List[str]],
):
    selected = list(selected or [])
    if ctx.triggered_id == "explorer-clear-btn":
        selected = []

    if not MONTHS or EXPLORER["monthly"].empty:
        return [], selected, f"Selection ({len(selected)})"

    end_month = MONTHS[min(slider_val[1], len(MONTHS) - 1)]
    snap = EXPLORER["monthly"][EXPLORER["monthly"]["month"] == end_month].copy()

    filtered = snap.copy()
    if map_scope in SCOPE_TO_CONTINENT:
        filtered = filtered[(~filtered["is_aggregate"]) & (filtered["continent"] == SCOPE_TO_CONTINENT[map_scope])]
    elif type_filter == "COUNTRIES":
        filtered = filtered[~filtered["is_aggregate"]]
    elif type_filter == "AGGREGATES":
        filtered = filtered[filtered["is_aggregate"]]
    elif isinstance(type_filter, str) and type_filter.startswith("CONT::"):
        cont_name = type_filter.split("::", 1)[1]
        filtered = filtered[(~filtered["is_aggregate"]) & (filtered["continent"] == cont_name)]

    if search_text and search_text.strip():
        q = search_text.strip().lower()
        filtered = filtered[
            filtered["location"].str.lower().str.contains(q, na=False)
            | filtered["iso_code"].str.lower().str.contains(q, na=False)
        ]

    sort_col = metric if sort_by == "metric" else sort_by
    asc = sort_dir == "asc"
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=asc, na_position="last")

    sel_df = snap[snap["entity_id"].isin(selected)].copy()
    if sort_col in sel_df.columns:
        sel_df = sel_df.sort_values(sort_col, ascending=asc, na_position="last")

    rest_df = filtered[~filtered["entity_id"].isin(selected)]
    combined = pd.concat([sel_df, rest_df], ignore_index=True)

    max_abs = float(snap[metric].abs().quantile(0.98)) if metric in snap.columns and snap[metric].notna().any() else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    options = []
    for _, row in combined.iterrows():
        val = row[metric] if metric in row else np.nan
        val_str = format_compact_val(val, metric)
        bar_pct = 0 if pd.isna(val) else min(100, int(abs(float(val)) / max_abs * 100))
        is_sel = row["entity_id"] in selected
        bg = (
            f"linear-gradient(90deg, rgba(29,78,216,0.14) {bar_pct}%, transparent {bar_pct}%)"
            if is_sel
            else f"linear-gradient(90deg, rgba(148,163,184,0.16) {bar_pct}%, transparent {bar_pct}%)"
        )

        # Фиксированная ширина 242px гарантирует аккуратный разрыв между страной и числом в Dash 4
        options.append(
            {
                "label": html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "242px",
                        "gap": "10px",
                        "padding": "3px 6px",
                        "borderRadius": "4px",
                        "background": bg,
                    },
                    children=[
                        html.Span(
                            row["location"],
                            style={
                                "fontWeight": "600" if is_sel else "400",
                                "color": "#0f172a",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        html.Span(
                            val_str,
                            style={
                                "color": "#64748b" if val_str == "No data" else "#1e293b",
                                "fontVariantNumeric": "tabular-nums",
                                "flexShrink": "0",
                                "marginLeft": "auto",
                            },
                        ),
                    ],
                ),
                "value": row["entity_id"],
            }
        )

    scope_note = f" — {SCOPE_TO_CONTINENT[map_scope]}" if map_scope in SCOPE_TO_CONTINENT else ""
    return options, selected, f"Selection ({len(selected)}){scope_note}"


def _filter_map_countries(snap_df: pd.DataFrame, metric: str, selected: List[str], only_selected: bool) -> pd.DataFrame:
    cdf = snap_df[(~snap_df["is_aggregate"]) & (snap_df[metric].notna())].copy()
    selected_isos = [s for s in selected if not str(s).startswith("AGG::")]
    if only_selected and selected_isos:
        cdf = cdf[cdf["entity_id"].isin(selected_isos)]
    return cdf


def build_explorer_choropleth(
    snap_df: pd.DataFrame, metric: str, scope: str, title: str, selected: List[str], only_selected: bool
) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    cdf = _filter_map_countries(snap_df, metric, selected, only_selected)
    if cdf.empty:
        return empty_figure(f"Нет данных по выбранным странам для «{meta['label']}» на эту дату")

    zmin = float(cdf[metric].quantile(0.02)) if len(cdf) > 2 else float(cdf[metric].min())
    zmax = float(cdf[metric].quantile(0.98)) if len(cdf) > 2 else float(cdf[metric].max())
    if zmin == zmax:
        zmin, zmax = min(0.0, zmin), max(zmax * 1.1, 1.0)
    if meta["diverging_zero"]:
        bound = max(abs(zmin), abs(zmax), 1.0)
        zmin, zmax = -bound, bound

    fig = go.Figure(
        data=go.Choropleth(
            locations=cdf["iso_code"],
            z=cdf[metric],
            text=cdf["location"],
            colorscale=meta["colorscale"],
            zmin=zmin,
            zmax=zmax,
            marker_line_color="#ffffff",
            marker_line_width=0.6,
            colorbar=dict(title=meta["unit"], orientation="h", y=-0.08, thickness=12, len=0.65),
            hovertemplate="<b>%{text}</b><br>" + f"{meta['label']}: %{{z:,.2f}} {meta['unit']}<extra></extra>",
        )
    )
    fig.update_geos(
        scope=scope,
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#cbd5e1",
        showland=True,
        landcolor="#eceff1",
        showcountries=True,
        countrycolor="#cbd5e1",
        projection_type="natural earth" if scope == "world" else "mercator",
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        margin=dict(l=10, r=10, t=42, b=20),
        height=480,
        template="plotly_white",
    )
    return fig


def build_compare_two_maps(
    snap1: pd.DataFrame,
    snap2: pd.DataFrame,
    metric: str,
    scope: str,
    m1: str,
    m2: str,
    selected: List[str],
    only_selected: bool,
) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    c1 = _filter_map_countries(snap1, metric, selected, only_selected)
    c2 = _filter_map_countries(snap2, metric, selected, only_selected)
    both_vals = pd.concat([c1[metric], c2[metric]]).dropna()
    if both_vals.empty:
        return empty_figure(f"Нет данных по выбранным странам для сравнения ({m1} vs {m2})")

    zmin = float(both_vals.quantile(0.02)) if len(both_vals) > 2 else float(both_vals.min())
    zmax = float(both_vals.quantile(0.98)) if len(both_vals) > 2 else float(both_vals.max())
    if zmin == zmax:
        zmin, zmax = min(0.0, zmin), max(zmax * 1.1, 1.0)
    if meta["diverging_zero"]:
        bound = max(abs(zmin), abs(zmax), 1.0)
        zmin, zmax = -bound, bound

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "geo"}, {"type": "geo"}]],
        subplot_titles=[f"Дата 1: {m1}", f"Дата 2: {m2}"],
        horizontal_spacing=0.03,
    )

    for col_idx, cdf, show_cb in [(1, c1, False), (2, c2, True)]:
        fig.add_trace(
            go.Choropleth(
                locations=cdf["iso_code"],
                z=cdf[metric],
                text=cdf["location"],
                colorscale=meta["colorscale"],
                zmin=zmin,
                zmax=zmax,
                showscale=show_cb,
                marker_line_color="#ffffff",
                marker_line_width=0.5,
                colorbar=dict(title=meta["unit"], orientation="h", y=-0.08, x=0.5, thickness=12, len=0.6) if show_cb else None,
                hovertemplate="<b>%{text}</b><br>" + f"{meta['label']}: %{{z:,.2f}} {meta['unit']}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

    fig.update_geos(
        scope=scope,
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#cbd5e1",
        showland=True,
        landcolor="#eceff1",
        showcountries=True,
        countrycolor="#cbd5e1",
        projection_type="natural earth" if scope == "world" else "mercator",
    )
    fig.update_layout(margin=dict(l=8, r=8, t=42, b=30), height=480, template="plotly_white")
    return fig


def build_explorer_line(
    selected: List[str], metric: str, start_m: str, end_m: str, show_rangeslider: bool = False
) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    if not selected:
        return empty_figure("Выберите хотя бы одну страну или регион в правой панели")

    wdf = EXPLORER["weekly"]
    ent = EXPLORER["entities"].set_index("entity_id")["location"].to_dict()
    sub = wdf[
        (wdf["entity_id"].isin(selected))
        & (wdf["week_date"] >= f"{start_m}-01")
        & (wdf["week_date"] <= f"{end_m}-28")
        & (wdf[metric].notna())
    ].copy()

    if sub.empty:
        return empty_figure("Нет непустых точек для выбранных стран за этот период")

    fig = go.Figure()
    annotations = []
    for idx, eid in enumerate(selected):
        g = sub[sub["entity_id"] == eid].sort_values("week_date")
        if g.empty:
            continue
        name = ent.get(eid, eid)
        color = OWID_PALETTE[idx % len(OWID_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=g["week_date"],
                y=g[metric],
                mode="lines",
                name=name,
                line=dict(color=color, width=2.3),
                hovertemplate=f"<b>{name}</b>: %{{y:,.2f}} {meta['unit']}<extra></extra>",
            )
        )
        annotations.append(
            dict(
                x=g["week_date"].iloc[-1],
                y=float(g[metric].iloc[-1]),
                xref="x",
                yref="y",
                text=f" {name}",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color=color),
            )
        )

    if meta["diverging_zero"]:
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        annotations=annotations,
        yaxis_title=meta["unit"],
        xaxis=dict(rangeslider=dict(visible=show_rangeslider)),
        margin=dict(l=55, r=125, t=24, b=40),
        height=470,
    )
    return fig


def build_year_overlay(selected: List[str], metric: str) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    target_id = selected[0] if selected else "KAZ"
    ent = EXPLORER["entities"].set_index("entity_id")["location"].to_dict()
    target_name = ent.get(target_id, target_id)

    wdf = EXPLORER["weekly"]
    sub = wdf[(wdf["entity_id"] == target_id) & (wdf[metric].notna())].copy()
    if sub.empty:
        return empty_figure(f"Нет данных по годам для {target_name}")

    sub["year"] = sub["week_date"].dt.year
    sub["day_of_year"] = sub["week_date"].dt.dayofyear
    sub["pseudo_date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(sub["day_of_year"] - 1, unit="D")

    year_colors = {2020: "#b91c1c", 2021: "#7e22ce", 2022: "#0f766e", 2023: "#1d4ed8", 2024: "#64748b"}
    fig = go.Figure()
    annotations = []

    for yr in sorted(sub["year"].unique()):
        if yr < 2020 or yr > 2024:
            continue
        g = sub[sub["year"] == yr].sort_values("pseudo_date")
        if g.empty:
            continue
        color = year_colors.get(yr, "#334155")
        fig.add_trace(
            go.Scatter(
                x=g["pseudo_date"],
                y=g[metric],
                mode="lines",
                name=str(yr),
                line=dict(color=color, width=2.3),
                hovertemplate=f"<b>{yr}</b> (%{{x|%b %d}}): %{{y:,.2f}} {meta['unit']}<extra></extra>",
            )
        )
        annotations.append(
            dict(
                x=g["pseudo_date"].iloc[-1],
                y=float(g[metric].iloc[-1]),
                text=f" {yr}",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color=color),
            )
        )

    fig.update_layout(
        title=dict(text=f"Сравнение годов (Jan–Dec): {target_name} — {meta['label']}", font=dict(size=14)),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        annotations=annotations,
        xaxis=dict(tickformat="%b", dtick="M1"),
        yaxis_title=meta["unit"],
        margin=dict(l=55, r=80, t=44, b=40),
        height=470,
    )
    return fig


def build_small_multiples(selected: List[str], start_m: str, end_m: str) -> go.Figure:
    """Сетка 2x3 по 6 метрикам с чистой легендой сверху (без наложения) и customdata для клик-зума."""
    sel = selected[:6] if selected else ["KAZ", "USA", "DEU"]
    ent = EXPLORER["entities"].set_index("entity_id")["location"].to_dict()
    wdf = EXPLORER["weekly"]
    sub = wdf[
        (wdf["entity_id"].isin(sel))
        & (wdf["week_date"] >= f"{start_m}-01")
        & (wdf["week_date"] <= f"{end_m}-28")
    ].copy()

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[t for _, t in FACET_METRICS],
        vertical_spacing=0.17,
        horizontal_spacing=0.07,
    )

    for m_idx, (col, title) in enumerate(FACET_METRICS):
        r, c = divmod(m_idx, 3)
        for s_idx, eid in enumerate(sel):
            g = sub[(sub["entity_id"] == eid) & (sub[col].notna())].sort_values("week_date")
            if g.empty:
                continue
            name = ent.get(eid, eid)
            fig.add_trace(
                go.Scatter(
                    x=g["week_date"],
                    y=g[col],
                    mode="lines",
                    name=name,
                    legendgroup=eid,
                    showlegend=(m_idx == 0),
                    customdata=[col] * len(g),
                    line=dict(color=OWID_PALETTE[s_idx % len(OWID_PALETTE)], width=2.0),
                    hovertemplate=f"<b>{name}</b> ({title})<br>%{{x|%Y-%m-%d}}: %{{y:,.2f}}<br><i>Клик — увеличить</i><extra></extra>",
                ),
                row=r + 1,
                col=c + 1,
            )

    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=48, r=20, t=92, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor="#f8fafc",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant",
        ),
    )
    return fig


def build_explorer_bar(snap_df: pd.DataFrame, selected: List[str], metric: str, scope: str, end_m: str) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    if len(selected) >= 2:
        bdf = snap_df[(snap_df["entity_id"].isin(selected)) & (snap_df[metric].notna())].copy()
        title = f"Сравнение выбранных стран/регионов на {end_m}"
    else:
        bdf = snap_df[(~snap_df["is_aggregate"]) & (snap_df[metric].notna())].copy()
        if scope in SCOPE_TO_CONTINENT:
            bdf = bdf[bdf["continent"] == SCOPE_TO_CONTINENT[scope]]
        bdf = bdf.sort_values(metric, ascending=False).head(15)
        title = f"Топ-15 стран на {end_m} ({SCOPE_TO_CONTINENT.get(scope, 'World')})"

    if bdf.empty:
        return empty_figure("Нет данных для столбчатой диаграммы на выбранную дату")

    bdf = bdf.sort_values(metric, ascending=True)
    colors = ["#dc2626" if v > 0 and meta["diverging_zero"] else "#2563eb" for v in bdf[metric]]
    fig = go.Figure(
        go.Bar(
            x=bdf[metric],
            y=bdf["location"],
            orientation="h",
            marker_color=colors,
            text=[format_compact_val(v, metric) for v in bdf[metric]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=meta["unit"],
        template="plotly_white",
        margin=dict(l=130, r=65, t=44, b=40),
        height=470,
    )
    return fig


@app.callback(
    Output("explorer-title", "children"),
    Output("explorer-subtitle", "children"),
    Output("explorer-start-label", "children"),
    Output("explorer-end-label", "children"),
    Output("explorer-graph", "figure"),
    Output("explorer-graph-wrapper", "style"),
    Output("explorer-table-wrapper", "children"),
    Output("explorer-table-wrapper", "style"),
    Output("explorer-facet-bar", "style"),
    Output("facet-grid-controls", "style"),
    Output("facet-zoom-controls", "style"),
    Output("facet-zoom-label", "children"),
    Input("explorer-metric", "value"),
    Input("explorer-view-mode", "value"),
    Input("explorer-slider", "value"),
    Input("explorer-map-scope", "value"),
    Input("explorer-countries", "value"),
    Input("explorer-map-only-selected", "value"),
    Input("explorer-facet-zoom", "data"),
)
def update_explorer_main(
    metric: str,
    view_mode: str,
    slider_val: List[int],
    map_scope: str,
    selected: Optional[List[str]],
    only_sel_vals: Optional[List[str]],
    facet_zoom: Optional[str],
):
    selected = list(selected or [])
    only_selected = bool(only_sel_vals and "only" in only_sel_vals)
    meta = EXPLORER_METRICS[metric]

    hide = {"display": "none"}
    if not MONTHS or EXPLORER["monthly"].empty:
        return (
            meta["label"], meta["subtitle"], "-", "-", empty_figure("Данные не загружены"),
            {"display": "block"}, "", hide, hide, hide, hide, "",
        )

    m1 = MONTHS[min(slider_val[0], len(MONTHS) - 1)]
    m2 = MONTHS[min(slider_val[1], len(MONTHS) - 1)]
    snap1 = EXPLORER["monthly"][EXPLORER["monthly"]["month"] == m1]
    snap2 = EXPLORER["monthly"][EXPLORER["monthly"]["month"] == m2]

    facet_bar_style = hide
    grid_ctrl_style = hide
    zoom_ctrl_style = hide
    zoom_label = ""

    if view_mode == "facets":
        facet_bar_style = {"display": "block", "marginBottom": "10px", "padding": "8px 10px", "backgroundColor": "#f8fafc", "borderRadius": "8px", "border": "1px solid #e2e8f0"}
        if facet_zoom and facet_zoom in EXPLORER_METRICS:
            grid_ctrl_style = hide
            zoom_ctrl_style = {"display": "flex", "alignItems": "center", "gap": "12px"}
            z_meta = EXPLORER_METRICS[facet_zoom]
            zoom_label = f"Увеличенный вид: {z_meta['label']}"
            title_str = f"{z_meta['label']} ({m1} — {m2})"
            subtitle_str = z_meta["subtitle"]
            fig = build_explorer_line(selected, facet_zoom, m1, m2, show_rangeslider=True)
            return title_str, subtitle_str, m1, m2, fig, {"display": "block"}, "", hide, facet_bar_style, grid_ctrl_style, zoom_ctrl_style, zoom_label
        else:
            grid_ctrl_style = {"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "6px"}
            zoom_ctrl_style = hide
            title_str = f"Small Multiples — 6 ключевых метрик ({m1} — {m2})"
            subtitle_str = "Сравнение выбранных стран сразу по 6 показателям. Нажмите на любой график или кнопку выше для увеличения."
            fig = build_small_multiples(selected, m1, m2)
            return title_str, subtitle_str, m1, m2, fig, {"display": "block"}, "", hide, facet_bar_style, grid_ctrl_style, zoom_ctrl_style, zoom_label

    if view_mode == "compare_map":
        title_str = f"{meta['label']} ({m1} vs {m2})"
    elif view_mode in ("map", "bar", "table"):
        title_str = f"{meta['label']} — {m2}"
    else:
        title_str = f"{meta['label']} ({m1} — {m2})"

    if view_mode == "table":
        t1 = snap1[["entity_id", metric]].rename(columns={metric: f"val_{m1}"})
        t2 = snap2[["entity_id", "location", "continent", "population", "gdp_per_capita", metric]].rename(
            columns={metric: f"val_{m2}"}
        )
        tdf = t2.merge(t1, on="entity_id", how="left")
        if map_scope in SCOPE_TO_CONTINENT:
            tdf = tdf[tdf["continent"] == SCOPE_TO_CONTINENT[map_scope]]
        tdf["change"] = (tdf[f"val_{m2}"] - tdf[f"val_{m1}"]).round(2)
        tdf[f"val_{m1}"] = tdf[f"val_{m1}"].round(2)
        tdf[f"val_{m2}"] = tdf[f"val_{m2}"].round(2)
        tdf = tdf.sort_values(f"val_{m2}", ascending=False, na_position="last")
        cols = ["location", "continent", f"val_{m1}", f"val_{m2}", "change"]
        table_el = dash_table.DataTable(
            data=tdf[cols].head(250).to_dict("records"),
            columns=[
                {"name": "Страна / Регион", "id": "location"},
                {"name": "Континент", "id": "continent"},
                {"name": f"Значение ({m1})", "id": f"val_{m1}"},
                {"name": f"Значение ({m2})", "id": f"val_{m2}"},
                {"name": "Изменение", "id": "change"},
            ],
            page_size=14,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "12px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
        )
        return title_str, meta["subtitle"], m1, m2, empty_figure(""), hide, table_el, {"display": "block"}, hide, hide, hide, ""

    if view_mode == "map":
        fig = build_explorer_choropleth(snap2, metric, map_scope, f"{meta['label']} ({m2})", selected, only_selected)
    elif view_mode == "compare_map":
        fig = build_compare_two_maps(snap1, snap2, metric, map_scope, m1, m2, selected, only_selected)
    elif view_mode == "line":
        fig = build_explorer_line(selected, metric, m1, m2)
    elif view_mode == "year_overlay":
        fig = build_year_overlay(selected, metric)
    else:
        fig = build_explorer_bar(snap2, selected, metric, map_scope, m2)

    return title_str, meta["subtitle"], m1, m2, fig, {"display": "block"}, "", hide, hide, hide, hide, ""


@app.callback(
    Output("explorer-download", "data"),
    Input("explorer-download-btn", "n_clicks"),
    State("explorer-metric", "value"),
    State("explorer-slider", "value"),
    State("explorer-countries", "value"),
    prevent_initial_call=True,
)
def download_explorer_csv(_n, metric: str, slider_val: List[int], selected: Optional[List[str]]):
    m1 = MONTHS[min(slider_val[0], len(MONTHS) - 1)]
    m2 = MONTHS[min(slider_val[1], len(MONTHS) - 1)]
    sub = EXPLORER["monthly"][
        (EXPLORER["monthly"]["month"] >= m1) & (EXPLORER["monthly"]["month"] <= m2)
    ][["iso_code", "location", "continent", "month", metric]].copy()
    if selected:
        sel_sub = EXPLORER["monthly"][
            (EXPLORER["monthly"]["entity_id"].isin(selected))
            & (EXPLORER["monthly"]["month"] >= m1)
            & (EXPLORER["monthly"]["month"] <= m2)
        ][["iso_code", "location", "continent", "month", metric]].copy()
        if not sel_sub.empty:
            sub = sel_sub
    return dcc.send_data_frame(sub.to_csv, f"owid_{metric}_{m1}_to_{m2}.csv", index=False)


# ==================== ВКЛАДКА 2: ОБЗОР SQLITE (ЕДИНЫЕ ЦВЕТА + ЛЕГЕНДА + ЧИСТЫЙ ЗАГОЛОВОК) ====================

@app.callback(
    Output("kpi-row", "children"),
    Output("overview-timeseries", "figure"),
    Output("overview-top-countries", "figure"),
    Input("filter-year", "value"),
    Input("filter-region", "value"),
    Input("filter-category", "value"),
)
def update_overview(year_filter: str, region: str, category: str):
    start_date, end_date = build_year_bounds(year_filter)
    meta = CATEGORY_META.get(category, CATEGORY_META["cases_pm"])
    table = meta["table"]
    metric_col = meta["per_million_col"]

    where_region = "" if region == "ALL" else "AND l.continent = ?"
    params = (start_date, end_date) if region == "ALL" else (start_date, end_date, region)

    kpi_df = query_sqlite(
        f"""
        SELECT
            COUNT(DISTINCT l.iso_code) AS countries,
            ROUND(SUM(COALESCE(e.new_cases, 0)), 0) AS cases_sum,
            ROUND(SUM(COALESCE(e.new_deaths, 0)), 0) AS deaths_sum
        FROM epidemiology e
        JOIN locations l ON e.iso_code = l.iso_code
        WHERE e.date BETWEEN ? AND ? {where_region}
        """,
        params,
    )
    vacc_df = query_sqlite(
        f"""
        SELECT ROUND(AVG(country_max), 1) AS avg_vacc
        FROM (
            SELECT v.iso_code, MAX(v.people_fully_vaccinated_per_hundred) AS country_max
            FROM vaccinations v
            JOIN locations l ON v.iso_code = l.iso_code
            WHERE v.date BETWEEN ? AND ? {where_region}
              AND v.people_fully_vaccinated_per_hundred IS NOT NULL
            GROUP BY v.iso_code
        )
        """,
        params,
    )

    countries_n = int(kpi_df["countries"].iloc[0]) if not kpi_df.empty and pd.notna(kpi_df["countries"].iloc[0]) else 0
    cases_sum = float(kpi_df["cases_sum"].iloc[0]) if not kpi_df.empty and pd.notna(kpi_df["cases_sum"].iloc[0]) else 0.0
    deaths_sum = float(kpi_df["deaths_sum"].iloc[0]) if not kpi_df.empty and pd.notna(kpi_df["deaths_sum"].iloc[0]) else 0.0
    avg_vacc = float(vacc_df["avg_vacc"].iloc[0]) if not vacc_df.empty and pd.notna(vacc_df["avg_vacc"].iloc[0]) else 0.0

    cards = [
        html.Div(style=CARD_STYLE, children=[html.Div("Стран в выборке", style={"fontSize": "12px", "color": "#64748b"}), html.Div(f"{countries_n}", style={"fontSize": "24px", "fontWeight": "700"})]),
        html.Div(style=CARD_STYLE, children=[html.Div("Новые случаи за период", style={"fontSize": "12px", "color": "#64748b"}), html.Div(f"{cases_sum:,.0f}", style={"fontSize": "24px", "fontWeight": "700", "color": "#1d4ed8"})]),
        html.Div(style=CARD_STYLE, children=[html.Div("Новые смерти за период", style={"fontSize": "12px", "color": "#64748b"}), html.Div(f"{deaths_sum:,.0f}", style={"fontSize": "24px", "fontWeight": "700", "color": "#b91c1c"})]),
        html.Div(style=CARD_STYLE, children=[html.Div("Средний макс. охват вакцинации", style={"fontSize": "12px", "color": "#64748b"}), html.Div(f"{avg_vacc:.1f}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#15803d"})]),
    ]

    ts_df = query_sqlite(
        f"""
        SELECT
            STRFTIME('%Y-%m', t.date) AS month,
            l.continent,
            ROUND(AVG(t.{metric_col}), 2) AS metric_val
        FROM {table} t
        JOIN locations l ON t.iso_code = l.iso_code
        WHERE t.date BETWEEN ? AND ? {where_region}
          AND t.{metric_col} IS NOT NULL
        GROUP BY month, l.continent
        ORDER BY month
        """,
        params,
    )
    if ts_df.empty:
        fig_ts = empty_figure("Нет данных для выбранных фильтров", height=440)
    else:
        fig_ts = px.line(
            ts_df,
            x="month",
            y="metric_val",
            color="continent",
            color_discrete_map=CONTINENT_COLORS,
            title=f"Помесячная динамика<br><sup>{meta['label']}</sup>",
            labels={"month": "Месяц", "metric_val": meta["unit"], "continent": "Континент"},
            template="plotly_white",
        )
        fig_ts.update_layout(
            hovermode="x unified",
            height=450,
            margin=dict(l=55, r=20, t=62, b=75),
            legend=dict(orientation="h", y=-0.24, x=0, title_text="Континент: "),
        )

    agg_sql = "MAX" if meta["agg"] == "max" else "AVG"
    top_df = query_sqlite(
        f"""
        SELECT l.location, l.continent, ROUND({agg_sql}(t.{metric_col}), 2) AS metric_val
        FROM {table} t
        JOIN locations l ON t.iso_code = l.iso_code
        WHERE t.date BETWEEN ? AND ? {where_region}
          AND l.population >= 1000000
          AND t.{metric_col} IS NOT NULL
        GROUP BY l.iso_code, l.location, l.continent
        ORDER BY metric_val DESC
        LIMIT 10
        """,
        params,
    )
    if top_df.empty:
        fig_top = empty_figure("Нет стран для выбранного фильтра", height=440)
    else:
        top_df = top_df.sort_values("metric_val", ascending=True)
        fig_top = px.bar(
            top_df,
            x="metric_val",
            y="location",
            color="continent",
            color_discrete_map=CONTINENT_COLORS,
            orientation="h",
            text="metric_val",
            title=f"Топ-10 стран (население ≥ 1 млн)<br><sup>{meta['label']} — цвет обозначает континент</sup>",
            labels={"metric_val": meta["unit"], "location": "", "continent": "Континент"},
            template="plotly_white",
        )
        fig_top.update_traces(texttemplate="%{text:,.1f}", textposition="outside", cliponaxis=False)
        fig_top.update_layout(
            height=450,
            showlegend=True,
            margin=dict(l=125, r=48, t=62, b=75),
            yaxis=dict(categoryorder="total ascending"),
            legend=dict(orientation="h", y=-0.24, x=0, title_text="Континент: "),
        )

    return cards, fig_ts, fig_top


# ==================== ВКЛАДКА 3: ДЕТЕКЦИЯ ВОЛН (СТРАНЫ + РЕГИОНЫ + СУММАРНО) ====================

def resolve_wave_isos(preset_key: str, manual_isos: Optional[List[str]]) -> Tuple[List[str], str]:
    loc_df = ARTIFACTS["locations"]
    if preset_key == "CUSTOM":
        isos = list(manual_isos or ["KAZ"])
        if len(isos) == 1:
            row = loc_df[loc_df["iso_code"] == isos[0]]
            label = f"{row['location'].iloc[0]} ({isos[0]})" if not row.empty else isos[0]
        else:
            label = f"Выбранные страны ({len(isos)})"
        return isos, label

    preset = WAVE_REGION_PRESETS.get(preset_key, {})
    if preset.get("isos") == "ALL":
        return loc_df["iso_code"].tolist(), "Весь мир (Global)"
    if preset.get("isos"):
        valid = loc_df[loc_df["iso_code"].isin(preset["isos"])]["iso_code"].tolist()
        return valid, preset["label"]
    if preset.get("continents"):
        valid = loc_df[loc_df["continent"].isin(preset["continents"])]["iso_code"].tolist()
        return valid, preset["label"]
    return ["KAZ"], "Kazakhstan (KAZ)"


def detect_peaks_on_series(series: pd.Series) -> pd.DataFrame:
    """Единый алгоритм детекции волн (как в src/analytics.py) для любого агрегированного ряда."""
    s = series.asfreq("D").interpolate(limit=14).fillna(0.0).astype("float64")
    values = s.values
    max_val = float(values.max()) if len(values) > 0 else 0.0
    if max_val < 10.0:
        return pd.DataFrame(columns=["wave_number", "peak_date", "peak_cases_pm_smoothed", "prominence"])

    min_height = max(15.0, 0.10 * max_val)
    min_prominence = max(10.0, 0.12 * max_val)
    peaks, props = find_peaks(values, distance=45, height=min_height, prominence=min_prominence)
    rows = []
    for idx, p in enumerate(peaks, start=1):
        rows.append(
            {
                "wave_number": idx,
                "peak_date": str(s.index[p].date()),
                "peak_cases_pm_smoothed": round(float(values[p]), 2),
                "prominence": round(float(props["prominences"][idx - 1]), 2),
            }
        )
    return pd.DataFrame(rows)


@app.callback(
    Output("wave-scope-badge", "children"),
    Output("waves-graph", "figure"),
    Output("waves-table-container", "children"),
    Input("wave-preset", "value"),
    Input("wave-countries", "value"),
    Input("wave-agg-mode", "value"),
)
def update_waves(preset_key: str, manual_isos: Optional[List[str]], agg_mode: str):
    isos, group_label = resolve_wave_isos(preset_key, manual_isos)
    if not isos:
        return "Страны не выбраны.", empty_figure("Выберите регион или хотя бы одну страну"), html.P("Нет данных.")

    loc_df = ARTIFACTS["locations"]
    sub_locs = loc_df[loc_df["iso_code"].isin(isos)].copy()
    names_preview = ", ".join(sub_locs["location"].head(8).tolist())
    if len(sub_locs) > 8:
        names_preview += f" и ещё {len(sub_locs) - 8}"

    placeholders = ",".join(["?"] * len(isos))
    ts_df = query_sqlite(
        f"""
        SELECT e.iso_code, e.date, e.new_cases_smoothed, e.new_cases_per_million, l.location, l.population
        FROM epidemiology e
        JOIN locations l ON e.iso_code = l.iso_code
        WHERE e.iso_code IN ({placeholders})
          AND e.date BETWEEN '2020-03-01' AND '2023-12-31'
        ORDER BY e.date
        """,
        tuple(isos),
    )
    if ts_df.empty:
        return f"Охват: {names_preview}", empty_figure(f"Нет данных по выборке {group_label}"), html.P("Волны не найдены.")

    ts_df["date"] = pd.to_datetime(ts_df["date"])

    # Режим 1: одна страна ИЛИ суммарная волна по региону / группе стран
    if agg_mode == "aggregate" or len(isos) == 1:
        total_pop = float(sub_locs["population"].dropna().sum())
        if len(isos) == 1:
            iso0 = isos[0]
            if total_pop > 0 and ts_df["new_cases_smoothed"].notna().sum() > 30:
                s = ts_df.set_index("date")["new_cases_smoothed"] / total_pop * 1e6
            else:
                s = ts_df.set_index("date")["new_cases_per_million"].rolling(7, min_periods=1).mean()
            waves_sub = ARTIFACTS["waves"][ARTIFACTS["waves"]["iso_code"] == iso0][
                ["wave_number", "peak_date", "peak_cases_pm_smoothed", "prominence"]
            ].copy()
            if waves_sub.empty:
                waves_sub = detect_peaks_on_series(s)
        else:
            daily_sum = ts_df.groupby("date")["new_cases_smoothed"].sum()
            if total_pop > 0:
                s = daily_sum / total_pop * 1e6
            else:
                s = ts_df.groupby("date")["new_cases_per_million"].mean()
            waves_sub = detect_peaks_on_series(s)

        s = s.sort_index()
        y_max = float(s.max()) if len(s) > 0 and pd.notna(s.max()) else 10.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name="7-дневное сглаживание (случаи / млн)",
                line=dict(color="#1d4ed8", width=2.2),
            )
        )
        if not waves_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(waves_sub["peak_date"]),
                    y=waves_sub["peak_cases_pm_smoothed"],
                    mode="markers+text",
                    name=f"Пики волн ({len(waves_sub)})",
                    text=[f"W{n}" for n in waves_sub["wave_number"]],
                    textposition="top center",
                    marker=dict(color="#dc2626", size=10, symbol="diamond"),
                )
            )
        fig.update_layout(
            title=f"{group_label} — обнаружено волн: {len(waves_sub)}",
            xaxis_title="Дата",
            yaxis_title="Новые случаи на 1 млн (7d сглаж.)",
            yaxis=dict(range=[0, y_max * 1.18]),
            template="plotly_white",
            xaxis=dict(rangeslider=dict(visible=True)),
            height=470,
        )

        if waves_sub.empty:
            table = html.P("Для выбранной группы выраженных волн не обнаружено.")
        else:
            show_cols = ["wave_number", "peak_date", "peak_cases_pm_smoothed", "prominence"]
            table = dash_table.DataTable(
                data=waves_sub[show_cols].to_dict("records"),
                columns=[{"name": c, "id": c} for c in show_cols],
                style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
            )
        badge = f"Охват: {len(sub_locs)} стран ({names_preview}) | Суммарное население: {total_pop:,.0f}"
        return badge, fig, table

    # Режим 2: сравнение стран отдельно (до 8 самых крупных стран выборки, чтобы график оставался читаемым)
    top_compare = sub_locs.sort_values("population", ascending=False).head(8)["iso_code"].tolist()
    fig = go.Figure()
    all_peaks = []
    global_ymax = 10.0

    for idx, iso in enumerate(top_compare):
        sub_c = ts_df[ts_df["iso_code"] == iso].sort_values("date")
        if sub_c.empty:
            continue
        loc_name = str(sub_c["location"].iloc[0])
        pop = float(sub_c["population"].iloc[0]) if pd.notna(sub_c["population"].iloc[0]) else 0.0
        s = (
            sub_c.set_index("date")["new_cases_smoothed"] / pop * 1e6
            if pop > 0 and sub_c["new_cases_smoothed"].notna().sum() > 30
            else sub_c.set_index("date")["new_cases_per_million"].rolling(7, min_periods=1).mean()
        )
        if len(s) > 0 and pd.notna(s.max()):
            global_ymax = max(global_ymax, float(s.max()))
        color = OWID_PALETTE[idx % len(OWID_PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=s.index, y=s.values, mode="lines", name=loc_name,
                line=dict(color=color, width=2.0),
            )
        )
        w_sub = ARTIFACTS["waves"][ARTIFACTS["waves"]["iso_code"] == iso].copy()
        if not w_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(w_sub["peak_date"]),
                    y=w_sub["peak_cases_pm_smoothed"],
                    mode="markers",
                    name=f"{loc_name} (пики)",
                    showlegend=False,
                    marker=dict(color=color, size=8, symbol="diamond", line=dict(color="#0f172a", width=1)),
                )
            )
            all_peaks.append(w_sub[["location", "wave_number", "peak_date", "peak_cases_pm_smoothed", "prominence"]])

    fig.update_layout(
        title=f"Сравнение волн по странам — {group_label} (показано до 8 крупнейших стран)",
        xaxis_title="Дата",
        yaxis_title="Новые случаи на 1 млн (7d сглаж.)",
        yaxis=dict(range=[0, global_ymax * 1.15]),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True)),
        height=480,
    )

    if not all_peaks:
        table = html.P("Пики волн не обнаружены.")
    else:
        peaks_df = pd.concat(all_peaks, ignore_index=True).sort_values(["location", "wave_number"])
        table = dash_table.DataTable(
            data=peaks_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in peaks_df.columns],
            page_size=12,
            sort_action="native",
            style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
        )
    badge = f"Режим сравнения: показано {len(top_compare)} из {len(sub_locs)} стран ({names_preview})"
    return badge, fig, table


# ==================== ВКЛАДКА 4: ВАКЦИНАЦИЯ И ПРОГНОЗ (ФОКУС НА ФАКТ + ЛУЧШАЯ МОДЕЛЬ) ====================

@app.callback(
    Output("vacc-mortality-scatter", "figure"),
    Input("filter-region", "value"),
)
def update_vacc_scatter(region: str):
    where_region = "" if region == "ALL" else "AND l.continent = ?"
    params = () if region == "ALL" else (region,)
    df = query_sqlite(
        f"""
        WITH epi AS (
            SELECT iso_code, MAX(total_deaths_per_million) AS deaths_pm, MAX(total_cases_per_million) AS cases_pm
            FROM epidemiology GROUP BY iso_code
        ),
        vacc AS (
            SELECT iso_code, MAX(people_fully_vaccinated_per_hundred) AS full_vacc_pct
            FROM vaccinations GROUP BY iso_code
        )
        SELECT
            l.location, l.continent, l.population, l.gdp_per_capita,
            ROUND(e.deaths_pm, 1) AS deaths_pm,
            ROUND(e.deaths_pm * 100.0 / NULLIF(e.cases_pm, 0), 2) AS cfr_percent,
            ROUND(v.full_vacc_pct, 1) AS full_vacc_pct
        FROM locations l
        JOIN epi e ON l.iso_code = e.iso_code
        JOIN vacc v ON l.iso_code = v.iso_code
        WHERE l.population >= 1000000
          AND e.deaths_pm IS NOT NULL
          AND v.full_vacc_pct IS NOT NULL
          AND e.cases_pm > 0
          {where_region}
        """,
        params,
    )
    if df.empty:
        return empty_figure("Нет данных для графика вакцинации и летальности")

    fig = px.scatter(
        df, x="full_vacc_pct", y="cfr_percent", size="population", color="continent",
        color_discrete_map=CONTINENT_COLORS,
        hover_name="location", hover_data={"deaths_pm": True, "gdp_per_capita": ":.0f", "population": ":,.0f"},
        title="Вакцинация (%) против Летальности CFR (%) по странам (население ≥ 1 млн)",
        labels={"full_vacc_pct": "Полная вакцинация (% населения)", "cfr_percent": "Летальность CFR (%)", "continent": "Континент"},
        template="plotly_white", size_max=42,
    )
    fig.update_layout(height=430)
    return fig


@app.callback(
    Output("forecast-graph", "figure"),
    Output("forecast-table-container", "children"),
    Input("forecast-series", "value"),
    Input("forecast-focus-model", "value"),
)
def update_forecast(series_code: str, focus_choice: str):
    preds = ARTIFACTS["forecast_preds"]
    metrics = ARTIFACTS["forecast_metrics"]
    if preds.empty or metrics.empty:
        return empty_figure("Сначала запустите python -m src.forecasting"), html.P("Нет данных прогноза.")

    sub_p = preds[preds["series_code"] == series_code].copy()
    sub_m = metrics[metrics["series_code"] == series_code].sort_values("mae").copy()
    if sub_p.empty:
        return empty_figure(f"Нет прогноза для {series_code}"), html.P("Нет данных.")

    sub_p["date"] = pd.to_datetime(sub_p["date"])
    best_model = str(sub_m.iloc[0]["model"])
    focused_model = best_model if focus_choice == "AUTO_BEST" else focus_choice

    # 1. Сначала рисуем приглушённые (faded) второстепенные модели на заднем плане
    faded_colors = {
        "Linear Regression (AR lags)": "#dc2626",
        "Naive (baseline)": "#d97706",
        "Moving Average 7d": "#16a34a",
        "Seasonal Naive 7d (baseline)": "#7c3aed",
    }
    faded_dashes = {
        "Naive (baseline)": "dash",
        "Moving Average 7d": "dashdot",
        "Seasonal Naive 7d (baseline)": "dot",
        "Linear Regression (AR lags)": "dash",
    }

    fig = go.Figure()
    for model_name in sub_m["model"].tolist():
        if model_name == focused_model:
            continue
        grp = sub_p[sub_p["model"] == model_name]
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["predicted"],
                mode="lines",
                name=f"{model_name} (фоновая)",
                opacity=0.28,
                line=dict(color=faded_colors.get(model_name, "#64748b"), width=1.5, dash=faded_dashes.get(model_name, "dot")),
            )
        )

    # 2. Затем рисуем яркую выбранную/лучшую модель
    focus_grp = sub_p[sub_p["model"] == focused_model]
    if not focus_grp.empty:
        star = " ★ лучшая" if focused_model == best_model else " (в фокусе)"
        fig.add_trace(
            go.Scatter(
                x=focus_grp["date"],
                y=focus_grp["predicted"],
                mode="lines",
                name=f"{focused_model}{star}",
                opacity=1.0,
                line=dict(color="#dc2626", width=2.6),
            )
        )

    # 3. Поверх всего — чёткий чёрный Факт (actual)
    actual_df = sub_p[sub_p["model"] == best_model]
    fig.add_trace(
        go.Scatter(
            x=actual_df["date"],
            y=actual_df["actual"],
            mode="lines",
            name="Факт (actual)",
            opacity=1.0,
            line=dict(color="#0f172a", width=3.0),
        )
    )

    series_title = str(sub_m["series_name"].iloc[0]) if not sub_m.empty else series_code
    fig.update_layout(
        title=f"Факт против прогноза (в фокусе: {focused_model}): {series_title}",
        xaxis_title="Дата (тестовое окно)",
        yaxis_title="Новые случаи (7d сглаж.)",
        hovermode="x unified",
        template="plotly_white",
        height=440,
    )

    show_cols = ["model", "role", "mae", "rmse", "nmae_pct", "fit_ms", "predict_ms", "peak_memory_kb"]
    table = dash_table.DataTable(
        data=sub_m[show_cols].to_dict("records"),
        columns=[{"name": c, "id": c} for c in show_cols],
        style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
        style_data_conditional=[
            {
                "if": {"filter_query": f'{{model}} = "{focused_model}"'},
                "backgroundColor": "#fef2f2",
                "fontWeight": "600",
            }
        ],
    )
    return fig, table


# ==================== ВКЛАДКА 5: SQL-ВИТРИНА ====================

@app.callback(
    Output("sql-info", "children"),
    Output("sql-table-container", "children"),
    Input("sql-selector", "value"),
)
def update_sql_view(query_name: str):
    if not query_name:
        return "SQL-результаты не найдены.", html.P("Сначала запустите python -m src.database.")
    csv_path = SQL_OUTPUT_DIR / f"{query_name}.csv"
    if not csv_path.exists():
        return f"Файл не найден: {csv_path.name}", html.P("Нет данных.")

    df = pd.read_csv(csv_path)
    preview = df.head(100)
    info = f"Запрос: {query_name} | Всего строк: {len(df):,} | Показано первых: {len(preview)}"
    table = dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=[{"name": c, "id": c} for c in preview.columns],
        page_size=15,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
    )
    return info, table


def launch_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    print(f"\n[Dashboard] Запуск OWID Explorer & Analytics: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    launch_dashboard()
