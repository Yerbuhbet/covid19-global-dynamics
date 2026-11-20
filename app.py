import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from plotly.subplots import make_subplots

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

# 14 метрик для OWID Explorer (все присутствуют в compact.csv -> Parquet)
EXPLORER_METRICS: Dict[str, Dict] = {
    "excess_mortality": {
        "label": "Excess mortality: P-score (% vs baseline)",
        "subtitle": "Процентная разница между зафиксированными и ожидаемыми смертями от всех причин (аналог OWID P-score).",
        "unit": "%",
        "fmt": "{:.1f}%",
        "colorscale": "RdYlBu_r",
        "diverging_zero": True,
        "agg": "last",
        "ffill_months": 6,
    },
    "excess_mortality_cumulative": {
        "label": "Excess mortality: Cumulative (% vs projection)",
        "subtitle": "Накопленная избыточная смертность в процентах относительно допандемийного прогноза.",
        "unit": "%",
        "fmt": "{:.1f}%",
        "colorscale": "RdYlBu_r",
        "diverging_zero": True,
        "agg": "last",
        "ffill_months": 6,
    },
    "excess_mortality_cumulative_per_million": {
        "label": "Excess mortality: Cumulative per million",
        "subtitle": "Накопленная избыточная смертность на 1 млн населения (идеально для сравнения 2 дат на карте).",
        "unit": "смертей / млн",
        "fmt": "{:,.0f}",
        "colorscale": "YlOrRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "excess_mortality_cumulative_absolute": {
        "label": "Excess mortality: Cumulative absolute deaths",
        "subtitle": "Абсолютное накопленное число избыточных смертей от всех причин с начала 2020 года.",
        "unit": "смертей",
        "fmt": "{:,.0f}",
        "colorscale": "YlOrRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "new_cases_smoothed_per_million": {
        "label": "Daily confirmed cases per million (7-day avg)",
        "subtitle": "Новые подтверждённые случаи COVID-19 на 1 млн населения (7-дневное сглаживание).",
        "unit": "случаев / млн",
        "fmt": "{:,.1f}",
        "colorscale": "Blues",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "new_deaths_smoothed_per_million": {
        "label": "Daily confirmed deaths per million (7-day avg)",
        "subtitle": "Новые подтверждённые смерти от COVID-19 на 1 млн населения (7-дневное сглаживание).",
        "unit": "смертей / млн",
        "fmt": "{:,.2f}",
        "colorscale": "Reds",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "total_cases_per_million": {
        "label": "Cumulative confirmed cases per million",
        "subtitle": "Суммарное число подтверждённых случаев на 1 млн жителей.",
        "unit": "случаев / млн",
        "fmt": "{:,.0f}",
        "colorscale": "Blues",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "total_deaths_per_million": {
        "label": "Cumulative confirmed deaths per million",
        "subtitle": "Суммарное число подтверждённых смертей от COVID-19 на 1 млн жителей.",
        "unit": "смертей / млн",
        "fmt": "{:,.0f}",
        "colorscale": "OrRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 6,
    },
    "people_fully_vaccinated_per_hundred": {
        "label": "Share of people fully vaccinated (%)",
        "subtitle": "Доля населения, завершившего первичный курс вакцинации против COVID-19.",
        "unit": "%",
        "fmt": "{:.1f}%",
        "colorscale": "Greens",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 12,
    },
    "total_boosters": {
        "label": "COVID-19 vaccine boosters administered (total)",
        "subtitle": "Общее число введённых бустерных доз вакцины (как на карте бустеров OWID).",
        "unit": "доз",
        "fmt": "{:,.0f}",
        "colorscale": "PuRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 12,
    },
    "total_boosters_per_hundred": {
        "label": "COVID-19 vaccine boosters per 100 people",
        "subtitle": "Число введённых бустерных доз на 100 человек населения.",
        "unit": "доз / 100 чел.",
        "fmt": "{:.1f}",
        "colorscale": "PuRd",
        "diverging_zero": False,
        "agg": "last",
        "ffill_months": 12,
    },
    "icu_patients_per_million": {
        "label": "COVID-19 patients in ICU per million",
        "subtitle": "Число пациентов с COVID-19 в отделениях интенсивной терапии на 1 млн человек.",
        "unit": "пациентов / млн",
        "fmt": "{:.1f}",
        "colorscale": "Purples",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "hosp_patients": {
        "label": "COVID-19 patients in hospital",
        "subtitle": "Текущее число госпитализированных пациентов с COVID-19.",
        "unit": "пациентов",
        "fmt": "{:,.0f}",
        "colorscale": "YlOrRd",
        "diverging_zero": False,
        "agg": "mean",
        "ffill_months": 1,
    },
    "stringency_index": {
        "label": "COVID-19 Stringency Index (0–100)",
        "subtitle": "Индекс жёсткости противоэпидемических мер (школы, транспорт, локдауны; 100 = максимум).",
        "unit": "баллов",
        "fmt": "{:.1f}",
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
        "label": "Заболеваемость (новые случаи / млн)",
        "table": "epidemiology",
        "per_million_col": "new_cases_per_million",
        "agg": "mean",
        "unit": "случаев / млн в день",
    },
    "deaths_pm": {
        "label": "Смертность (новые смерти / млн)",
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
        "label": "Жёсткость ограничений (Stringency Index)",
        "table": "hospital_policy",
        "per_million_col": "stringency_index",
        "agg": "mean",
        "unit": "баллов (0–100)",
    },
}

OWID_PALETTE = [
    "#1d4ed8", "#c2410c", "#15803d", "#7e22ce",
    "#0f766e", "#be123c", "#a16207", "#334155",
]


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

    # Справочник сущностей (страны + регионы)
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

    # Недельная таблица для точных графиков Line / Year Overlay / Facets
    df["week_date"] = df["date"].dt.to_period("W").dt.end_time.dt.normalize()
    agg_dict = {}
    for col in metric_cols:
        if col in df.columns:
            agg_dict[col] = "mean" if EXPLORER_METRICS[col]["agg"] == "mean" else "last"

    weekly = (
        df.sort_values("date")
        .groupby(["entity_id", "week_date"], as_index=False)
        .agg(agg_dict)
    )

    # Месячная таблица для мгновенной работы карт, баров, таблицы и ползунка
    df["month"] = df["date"].dt.strftime("%Y-%m")
    monthly = (
        df.sort_values("date")
        .groupby(["entity_id", "month"], as_index=False)
        .agg(agg_dict)
    )

    months = sorted(monthly["month"].unique().tolist())

    # Полная сетка (entity_id x month) с аккуратным ffill по лимиту метрики (как в OWID: ближайшее значение)
    full_idx = pd.MultiIndex.from_product([entities["entity_id"].tolist(), months], names=["entity_id", "month"])
    monthly_grid = monthly.set_index(["entity_id", "month"]).reindex(full_idx)
    monthly_filled = pd.DataFrame(index=full_idx)
    for col in metric_cols:
        if col in monthly_grid.columns:
            lim = EXPLORER_METRICS[col]["ffill_months"]
            monthly_filled[col] = monthly_grid[col].groupby(level=0).ffill(limit=lim)

    monthly_filled = monthly_filled.reset_index()
    monthly_filled = monthly_filled.merge(
        entities[["entity_id", "iso_code", "location", "continent", "is_aggregate", "population", "gdp_per_capita"]],
        on="entity_id",
        how="left",
    )

    return {
        "entities": entities,
        "weekly": weekly,
        "monthly": monthly_filled,
        "months": months,
    }


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

SLIDER_MARKS = {
    i: {"label": m[:4], "style": {"fontSize": "11px", "color": "#475569"}}
    for i, m in enumerate(MONTHS)
    if m.endswith("-01") or i == len(MONTHS) - 1
}

CONTINENTS = (
    sorted(ARTIFACTS["locations"]["continent"].dropna().unique().tolist())
    if not ARTIFACTS["locations"].empty
    else []
)
WAVE_COUNTRY_OPTIONS = (
    [
        {"label": f"{row['location']} ({row['iso_code']})", "value": row["iso_code"]}
        for _, row in ARTIFACTS["locations"][
            ARTIFACTS["locations"]["iso_code"].isin(ARTIFACTS["waves"]["iso_code"].unique())
        ].iterrows()
    ]
    if not ARTIFACTS["waves"].empty and not ARTIFACTS["locations"].empty
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
        # Шапка проекта
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
                                "gridTemplateColumns": "minmax(0, 1fr) 320px",
                                "gap": "16px",
                                "marginTop": "14px",
                                "alignItems": "start",
                            },
                            children=[
                                # Левая основная панель графика
                                html.Div(
                                    style=PANEL_STYLE,
                                    children=[
                                        # Заголовок и выбор показателя
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
                                        # Переключатель режимов (Table | Map | Compare 2 Dates | Line | Year Overlay | Facets | Bar) + Map Region
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
                                                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                                    children=[
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
                                                            style={"width": "155px"},
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
                                        # Область визуализации (график или таблица)
                                        html.Div(
                                            id="explorer-graph-wrapper",
                                            children=[dcc.Graph(id="explorer-graph", config={"displaylogo": False})],
                                        ),
                                        html.Div(id="explorer-table-wrapper", style={"display": "none"}),
                                        # Нижний таймлайн с кнопкой Play и двойным ползунком
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
                                        # Подвал графика + Download CSV
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
                                # Правая панель OWID: метрика + поиск + сортировка + чекбоксы стран
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
                                        # Сортировка + направление
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
                                        # Фильтр по типу (все / страны / континенты и доходы)
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
                                        # Счётчик выбранных + Clear
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
                                        # Прокручиваемый список стран со значениями
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
                                                        "padding": "4px 4px",
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
                # ==================== ВКЛАДКА 2: ОБЗОР SQLITE (3 ФИЛЬТРА) ====================
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
                            style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr", "gap": "16px"},
                            children=[
                                html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="overview-timeseries")]),
                                html.Div(style=PANEL_STYLE, children=[dcc.Graph(id="overview-top-countries")]),
                            ],
                        ),
                    ],
                ),
                # ==================== ВКЛАДКА 3: ДЕТЕКЦИЯ ВОЛН ====================
                dcc.Tab(
                    label="3. Детекция волн",
                    value="tab-waves",
                    children=[
                        html.Div(
                            style={**PANEL_STYLE, "marginTop": "14px"},
                            children=[
                                html.Div(
                                    style={"maxWidth": "360px", "marginBottom": "12px"},
                                    children=[
                                        html.Label("Выберите страну для анализа волн (217 стран):", style={"fontWeight": "600", "fontSize": "13px"}),
                                        dcc.Dropdown(
                                            id="wave-country",
                                            options=WAVE_COUNTRY_OPTIONS,
                                            value="KAZ",
                                            clearable=False,
                                        ),
                                    ],
                                ),
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
                                    style={"maxWidth": "380px", "marginBottom": "12px"},
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

    # Фильтр по типу / континенту
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

    # Всегда закрепляем выбранные страны сверху (как в OWID), чтобы поиск их не сбрасывал
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
        bg = f"linear-gradient(90deg, rgba(29,78,216,0.12) {bar_pct}%, transparent {bar_pct}%)" if is_sel else f"linear-gradient(90deg, rgba(148,163,184,0.16) {bar_pct}%, transparent {bar_pct}%)"

        options.append(
            {
                "label": html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                        "padding": "2px 4px",
                        "borderRadius": "4px",
                        "background": bg,
                    },
                    children=[
                        html.Span(row["location"], style={"fontWeight": "600" if is_sel else "400", "color": "#0f172a"}),
                        html.Span(
                            val_str,
                            style={"color": "#64748b" if val_str == "No data" else "#1e293b", "fontVariantNumeric": "tabular-nums"},
                        ),
                    ],
                ),
                "value": row["entity_id"],
            }
        )

    scope_note = f" — {SCOPE_TO_CONTINENT[map_scope]}" if map_scope in SCOPE_TO_CONTINENT else ""
    return options, selected, f"Selection ({len(selected)}){scope_note}"


def build_explorer_choropleth(snap_df: pd.DataFrame, metric: str, scope: str, title: str) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    cdf = snap_df[(~snap_df["is_aggregate"]) & (snap_df[metric].notna())].copy()
    if cdf.empty:
        return empty_figure(f"Нет данных по странам для «{meta['label']}» на выбранную дату")

    zmin = float(cdf[metric].quantile(0.02))
    zmax = float(cdf[metric].quantile(0.98))
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
            marker_line_width=0.5,
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
    snap1: pd.DataFrame, snap2: pd.DataFrame, metric: str, scope: str, m1: str, m2: str
) -> go.Figure:
    meta = EXPLORER_METRICS[metric]
    c1 = snap1[(~snap1["is_aggregate"]) & (snap1[metric].notna())].copy()
    c2 = snap2[(~snap2["is_aggregate"]) & (snap2[metric].notna())].copy()
    both_vals = pd.concat([c1[metric], c2[metric]]).dropna()
    if both_vals.empty:
        return empty_figure(f"Нет данных для сравнения двух дат ({m1} vs {m2})")

    zmin = float(both_vals.quantile(0.02))
    zmax = float(both_vals.quantile(0.98))
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
                marker_line_width=0.4,
                colorbar=dict(title=meta["unit"], orientation="h", y=-0.08, x=0.5, thickness=12, len=0.6) if show_cb else None,
                hovertemplate="<b>%{text}</b><br>" + f"{meta['label']}: %{{z:,.2f}} {meta['unit']}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

    geo_cfg = dict(
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
    fig.update_geos(**geo_cfg)
    fig.update_layout(
        margin=dict(l=8, r=8, t=42, b=30),
        height=480,
        template="plotly_white",
    )
    return fig


def build_explorer_line(selected: List[str], metric: str, start_m: str, end_m: str) -> go.Figure:
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
                line=dict(color=color, width=2.2),
                hovertemplate=f"<b>{name}</b>: %{{y:,.2f}} {meta['unit']}<extra></extra>",
            )
        )
        # Прямая подпись названия страны у правого края линии (стиль OWID)
        last_x = g["week_date"].iloc[-1]
        last_y = float(g[metric].iloc[-1])
        annotations.append(
            dict(
                x=last_x,
                y=last_y,
                xref="x",
                yref="y",
                text=f" {name}",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color=color, family="Inter, sans-serif"),
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
        xaxis_title="",
        margin=dict(l=55, r=125, t=20, b=40),
        height=470,
    )
    return fig


def build_year_overlay(selected: List[str], metric: str) -> go.Figure:
    """Наложение годов 2020–2023 друг на друга (Jan–Dec) для первой выбранной страны."""
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
    """Сетка 2x3 (Small Multiples) по 6 ключевым показателям для выбранных стран."""
    facet_metrics = [
        ("new_cases_smoothed_per_million", "Случаи / млн (7d)"),
        ("new_deaths_smoothed_per_million", "Смерти / млн (7d)"),
        ("people_fully_vaccinated_per_hundred", "Вакцинация (%)"),
        ("total_boosters_per_hundred", "Бустеры / 100 чел."),
        ("stringency_index", "Жёсткость мер (0–100)"),
        ("excess_mortality", "Избыт. смертность (%)"),
    ]
    sel = selected[:5] if selected else ["KAZ", "USA", "DEU"]
    ent = EXPLORER["entities"].set_index("entity_id")["location"].to_dict()
    wdf = EXPLORER["weekly"]
    sub = wdf[
        (wdf["entity_id"].isin(sel))
        & (wdf["week_date"] >= f"{start_m}-01")
        & (wdf["week_date"] <= f"{end_m}-28")
    ].copy()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[t for _, t in facet_metrics],
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
    )

    for m_idx, (col, _title) in enumerate(facet_metrics):
        r, c = divmod(m_idx, 3)
        for s_idx, eid in enumerate(sel):
            g = sub[(sub["entity_id"] == eid) & (sub[col].notna())].sort_values("week_date")
            if g.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=g["week_date"],
                    y=g[col],
                    mode="lines",
                    name=ent.get(eid, eid),
                    legendgroup=eid,
                    showlegend=(m_idx == 0),
                    line=dict(color=OWID_PALETTE[s_idx % len(OWID_PALETTE)], width=1.8),
                ),
                row=r + 1,
                col=c + 1,
            )

    fig.update_layout(
        template="plotly_white",
        height=490,
        margin=dict(l=45, r=20, t=42, b=35),
        legend=dict(orientation="h", y=1.09, x=0),
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
        margin=dict(l=130, r=60, t=44, b=40),
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
    Input("explorer-metric", "value"),
    Input("explorer-view-mode", "value"),
    Input("explorer-slider", "value"),
    Input("explorer-map-scope", "value"),
    Input("explorer-countries", "value"),
)
def update_explorer_main(
    metric: str,
    view_mode: str,
    slider_val: List[int],
    map_scope: str,
    selected: Optional[List[str]],
):
    selected = list(selected or [])
    meta = EXPLORER_METRICS[metric]
    if not MONTHS or EXPLORER["monthly"].empty:
        return meta["label"], meta["subtitle"], "-", "-", empty_figure("Данные не загружены"), {"display": "block"}, "", {"display": "none"}

    m1 = MONTHS[min(slider_val[0], len(MONTHS) - 1)]
    m2 = MONTHS[min(slider_val[1], len(MONTHS) - 1)]
    snap1 = EXPLORER["monthly"][EXPLORER["monthly"]["month"] == m1]
    snap2 = EXPLORER["monthly"][EXPLORER["monthly"]["month"] == m2]

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
        return title_str, meta["subtitle"], m1, m2, empty_figure(""), {"display": "none"}, table_el, {"display": "block"}

    if view_mode == "map":
        fig = build_explorer_choropleth(snap2, metric, map_scope, f"{meta['label']} ({m2})")
    elif view_mode == "compare_map":
        fig = build_compare_two_maps(snap1, snap2, metric, map_scope, m1, m2)
    elif view_mode == "line":
        fig = build_explorer_line(selected, metric, m1, m2)
    elif view_mode == "year_overlay":
        fig = build_year_overlay(selected, metric)
    elif view_mode == "facets":
        fig = build_small_multiples(selected, m1, m2)
    else:
        fig = build_explorer_bar(snap2, selected, metric, map_scope, m2)

    return title_str, meta["subtitle"], m1, m2, fig, {"display": "block"}, "", {"display": "none"}


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


# ==================== КОЛБЭКИ ВКЛАДОК 2–5 (SQLITE, ВОЛНЫ, ПРОГНОЗ, SQL) ====================

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
        fig_ts = empty_figure("Нет данных для выбранных фильтров", height=420)
    else:
        fig_ts = px.line(
            ts_df, x="month", y="metric_val", color="continent",
            title=f"Помесячная динамика: {meta['label']}",
            labels={"month": "Месяц", "metric_val": meta["unit"], "continent": "Континент"},
            template="plotly_white",
        )
        fig_ts.update_layout(hovermode="x unified", height=420, legend=dict(orientation="h", y=-0.22))

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
        fig_top = empty_figure("Нет стран для выбранного фильтра", height=420)
    else:
        top_df = top_df.sort_values("metric_val", ascending=True)
        fig_top = px.bar(
            top_df, x="metric_val", y="location", color="continent", orientation="h",
            title=f"Топ-10 стран (население >= 1 млн): {meta['label']}",
            labels={"metric_val": meta["unit"], "location": "Страна", "continent": "Континент"},
            template="plotly_white",
        )
        fig_top.update_layout(height=420, showlegend=False)

    return cards, fig_ts, fig_top


@app.callback(
    Output("waves-graph", "figure"),
    Output("waves-table-container", "children"),
    Input("wave-country", "value"),
)
def update_waves(iso_code: str):
    ts_df = query_sqlite(
        """
        SELECT date, new_cases_per_million, new_cases_smoothed
        FROM epidemiology
        WHERE iso_code = ? AND date BETWEEN '2020-03-01' AND '2023-12-31'
        ORDER BY date
        """,
        (iso_code,),
    )
    loc_row = ARTIFACTS["locations"][ARTIFACTS["locations"]["iso_code"] == iso_code]
    loc_name = str(loc_row["location"].iloc[0]) if not loc_row.empty else iso_code
    pop = float(loc_row["population"].iloc[0]) if not loc_row.empty and pd.notna(loc_row["population"].iloc[0]) else None

    if ts_df.empty:
        return empty_figure(f"Нет данных по стране {iso_code}"), html.P("Волны не найдены.")

    ts_df["date"] = pd.to_datetime(ts_df["date"])
    if pop and pop > 0 and ts_df["new_cases_smoothed"].notna().sum() > 30:
        ts_df["smoothed_pm"] = ts_df["new_cases_smoothed"] / pop * 1e6
    else:
        ts_df["smoothed_pm"] = ts_df["new_cases_per_million"].rolling(7, min_periods=1).mean()

    waves_sub = ARTIFACTS["waves"][ARTIFACTS["waves"]["iso_code"] == iso_code].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts_df["date"], y=ts_df["smoothed_pm"], mode="lines",
            name="7-дневное сглаживание (случаи / млн)", line=dict(color="#1d4ed8", width=2),
        )
    )
    if not waves_sub.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(waves_sub["peak_date"]), y=waves_sub["peak_cases_pm_smoothed"],
                mode="markers+text", name="Пики волн (find_peaks)",
                text=[f"W{n}" for n in waves_sub["wave_number"]], textposition="top center",
                marker=dict(color="#dc2626", size=10, symbol="diamond"),
            )
        )
    fig.update_layout(
        title=f"{loc_name} ({iso_code}) — обнаружено волн: {len(waves_sub)}",
        xaxis_title="Дата", yaxis_title="Новые случаи на 1 млн (7d сглаж.)",
        template="plotly_white", xaxis=dict(rangeslider=dict(visible=True)), height=460,
    )

    if waves_sub.empty:
        table = html.P("Для выбранной страны выраженных волн не обнаружено.")
    else:
        show_cols = ["wave_number", "peak_date", "peak_cases_pm_smoothed", "prominence"]
        table = dash_table.DataTable(
            data=waves_sub[show_cols].to_dict("records"),
            columns=[{"name": c, "id": c} for c in show_cols],
            style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
        )
    return fig, table


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
        hover_name="location", hover_data={"deaths_pm": True, "gdp_per_capita": ":.0f", "population": ":,.0f"},
        title="Вакцинация (%) против Летальности CFR (%) по странам (население >= 1 млн)",
        labels={"full_vacc_pct": "Полная вакцинация (% населения)", "cfr_percent": "Летальность CFR (%)", "continent": "Континент"},
        template="plotly_white", size_max=42,
    )
    fig.update_layout(height=430)
    return fig


@app.callback(
    Output("forecast-graph", "figure"),
    Output("forecast-table-container", "children"),
    Input("forecast-series", "value"),
)
def update_forecast(series_code: str):
    preds = ARTIFACTS["forecast_preds"]
    metrics = ARTIFACTS["forecast_metrics"]
    if preds.empty or metrics.empty:
        return empty_figure("Сначала запустите python -m src.forecasting"), html.P("Нет данных прогноза.")

    sub_p = preds[preds["series_code"] == series_code].copy()
    sub_m = metrics[metrics["series_code"] == series_code].sort_values("mae").copy()
    if sub_p.empty:
        return empty_figure(f"Нет прогноза для {series_code}"), html.P("Нет данных.")

    sub_p["date"] = pd.to_datetime(sub_p["date"])
    actual_df = sub_p[sub_p["model"] == "Naive (baseline)"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=actual_df["date"], y=actual_df["actual"], mode="lines", name="Факт (actual)", line=dict(color="#0f172a", width=2.5))
    )
    color_map = {
        "Linear Regression (AR lags)": "#dc2626",
        "Naive (baseline)": "#f59e0b",
        "Moving Average 7d": "#16a34a",
        "Seasonal Naive 7d (baseline)": "#7c3aed",
    }
    for model_name, grp in sub_p.groupby("model"):
        fig.add_trace(
            go.Scatter(x=grp["date"], y=grp["predicted"], mode="lines", name=model_name, line=dict(color=color_map.get(model_name, "#2563eb"), width=1.8))
        )

    series_title = str(sub_m["series_name"].iloc[0]) if not sub_m.empty else series_code
    fig.update_layout(
        title=f"Факт против прогноза моделей (без data leakage): {series_title}",
        xaxis_title="Дата (тестовое окно)", yaxis_title="Новые случаи (7d сглаж.)",
        hovermode="x unified", template="plotly_white", height=420,
    )

    show_cols = ["model", "role", "mae", "rmse", "nmae_pct", "fit_ms", "predict_ms", "peak_memory_kb"]
    table = dash_table.DataTable(
        data=sub_m[show_cols].to_dict("records"),
        columns=[{"name": c, "id": c} for c in show_cols],
        style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
    )
    return fig, table


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
        page_size=15, sort_action="native",
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
