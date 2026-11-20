import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from src.config import DB_PATH, METRICS_DIR, OUTPUTS_DIR

WAVES_PATH = METRICS_DIR / "waves_detected.csv"
FORECAST_METRICS_PATH = METRICS_DIR / "forecast_comparison.csv"
FORECAST_PREDS_PATH = METRICS_DIR / "forecast_predictions.csv"
SQL_OUTPUT_DIR = OUTPUTS_DIR / "sql"

CATEGORY_META: Dict[str, Dict[str, str]] = {
    "cases_pm": {
        "label": "Заболеваемость (новые случаи / млн)",
        "table": "epidemiology",
        "column": "new_cases_smoothed",
        "per_million_col": "new_cases_per_million",
        "agg": "mean",
        "unit": "случаев / млн в день",
    },
    "deaths_pm": {
        "label": "Смертность (новые смерти / млн)",
        "table": "epidemiology",
        "column": "new_deaths_smoothed",
        "per_million_col": "new_deaths_per_million",
        "agg": "mean",
        "unit": "смертей / млн в день",
    },
    "vacc_pct": {
        "label": "Вакцинация (полный курс, %)",
        "table": "vaccinations",
        "column": "people_fully_vaccinated_per_hundred",
        "per_million_col": "people_fully_vaccinated_per_hundred",
        "agg": "max",
        "unit": "% населения",
    },
    "stringency": {
        "label": "Жёсткость ограничений (Stringency Index)",
        "table": "hospital_policy",
        "column": "stringency_index",
        "per_million_col": "stringency_index",
        "agg": "mean",
        "unit": "баллов (0–100)",
    },
}


def query_sqlite(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Выполняет параметризованный SQL-запрос к covid.db с обработкой ошибок."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except sqlite3.Error as exc:
        print(f"[Dashboard SQL Error] {exc}")
        return pd.DataFrame()


def load_static_artifacts() -> Dict[str, pd.DataFrame]:
    """Загружает предрасчитанные артефакты (волны, прогноз, справочник стран, SQL-таблицы)."""
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


def build_year_bounds(year_filter: str) -> tuple:
    if year_filter == "ACTIVE":
        return "2020-01-01", "2023-12-31"
    if year_filter == "ALL":
        return "2020-01-01", "2026-12-31"
    return f"{year_filter}-01-01", f"{year_filter}-12-31"


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=15))
    fig.update_layout(template="plotly_white", xaxis=dict(visible=False), yaxis=dict(visible=False), height=380)
    return fig


ARTIFACTS = load_static_artifacts()
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

app = Dash(__name__, title="COVID-19 Global Dynamics Dashboard")

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #e2e8f0",
    "borderRadius": "10px",
    "padding": "14px 18px",
    "flex": "1",
    "minWidth": "180px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}

app.layout = html.Div(
    style={
        "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "backgroundColor": "#f8fafc",
        "minHeight": "100vh",
        "padding": "20px 28px",
        "color": "#0f172a",
    },
    children=[
        # Заголовок
        html.Div(
            style={"marginBottom": "18px"},
            children=[
                html.H1(
                    "COVID-19 Global Dynamics — Аналитический дашборд",
                    style={"margin": "0 0 6px 0", "fontSize": "26px", "color": "#0f172a"},
                ),
                html.P(
                    "Источник: Our World in Data (compact.csv) | Хранилище: SQLite (covid.db) + Parquet | Анализ волн, вакцинации и прогноза",
                    style={"margin": "0", "color": "#475569", "fontSize": "14px"},
                ),
            ],
        ),
        # Панель обязательных фильтров: Год, Регион, Категория
        html.Div(
            style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "backgroundColor": "#ffffff",
                "padding": "14px 18px",
                "borderRadius": "10px",
                "border": "1px solid #e2e8f0",
                "marginBottom": "18px",
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
        # Вкладки
        dcc.Tabs(
            id="main-tabs",
            value="tab-overview",
            children=[
                # Вкладка 1: Обзор и динамика
                dcc.Tab(
                    label="1. Обзор и динамика",
                    value="tab-overview",
                    children=[
                        html.Div(
                            id="kpi-row",
                            style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "margin": "16px 0"},
                        ),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr", "gap": "16px"},
                            children=[
                                html.Div(
                                    style={"backgroundColor": "#fff", "padding": "12px", "borderRadius": "10px", "border": "1px solid #e2e8f0"},
                                    children=[dcc.Graph(id="overview-timeseries")],
                                ),
                                html.Div(
                                    style={"backgroundColor": "#fff", "padding": "12px", "borderRadius": "10px", "border": "1px solid #e2e8f0"},
                                    children=[dcc.Graph(id="overview-top-countries")],
                                ),
                            ],
                        ),
                    ],
                ),
                # Вкладка 2: Детекция волн
                dcc.Tab(
                    label="2. Детекция волн",
                    value="tab-waves",
                    children=[
                        html.Div(
                            style={"backgroundColor": "#fff", "padding": "16px", "borderRadius": "10px", "border": "1px solid #e2e8f0", "marginTop": "16px"},
                            children=[
                                html.Div(
                                    style={"maxWidth": "360px", "marginBottom": "12px"},
                                    children=[
                                        html.Label("Выберите страну для анализа волн:", style={"fontWeight": "600", "fontSize": "13px"}),
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
                # Вкладка 3: Вакцинация и прогноз
                dcc.Tab(
                    label="3. Вакцинация и прогноз",
                    value="tab-vacc-forecast",
                    children=[
                        html.Div(
                            style={"backgroundColor": "#fff", "padding": "16px", "borderRadius": "10px", "border": "1px solid #e2e8f0", "marginTop": "16px"},
                            children=[
                                dcc.Graph(id="vacc-mortality-scatter"),
                            ],
                        ),
                        html.Div(
                            style={"backgroundColor": "#fff", "padding": "16px", "borderRadius": "10px", "border": "1px solid #e2e8f0", "marginTop": "16px"},
                            children=[
                                html.Div(
                                    style={"maxWidth": "360px", "marginBottom": "12px"},
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
                # Вкладка 4: SQL-витрина
                dcc.Tab(
                    label="4. SQL-витрина (covid.db)",
                    value="tab-sql",
                    children=[
                        html.Div(
                            style={"backgroundColor": "#fff", "padding": "16px", "borderRadius": "10px", "border": "1px solid #e2e8f0", "marginTop": "16px"},
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

    # 1. KPI-запрос из SQLite
    kpi_sql = f"""
        SELECT
            COUNT(DISTINCT l.iso_code) AS countries,
            ROUND(SUM(COALESCE(e.new_cases, 0)), 0) AS cases_sum,
            ROUND(SUM(COALESCE(e.new_deaths, 0)), 0) AS deaths_sum
        FROM epidemiology e
        JOIN locations l ON e.iso_code = l.iso_code
        WHERE e.date BETWEEN ? AND ? {where_region}
    """
    kpi_df = query_sqlite(kpi_sql, params)
    vacc_sql = f"""
        SELECT ROUND(AVG(country_max), 1) AS avg_vacc
        FROM (
            SELECT v.iso_code, MAX(v.people_fully_vaccinated_per_hundred) AS country_max
            FROM vaccinations v
            JOIN locations l ON v.iso_code = l.iso_code
            WHERE v.date BETWEEN ? AND ? {where_region}
              AND v.people_fully_vaccinated_per_hundred IS NOT NULL
            GROUP BY v.iso_code
        )
    """
    vacc_df = query_sqlite(vacc_sql, params)

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

    # 2. Временной ряд по месяцам (SQLite -> Pandas -> Plotly)
    ts_sql = f"""
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
    """
    ts_df = query_sqlite(ts_sql, params)
    if ts_df.empty:
        fig_ts = empty_figure("Нет данных для выбранных фильтров")
    else:
        fig_ts = px.line(
            ts_df,
            x="month",
            y="metric_val",
            color="continent",
            markers=False,
            title=f"Помесячная динамика: {meta['label']}",
            labels={"month": "Месяц", "metric_val": meta["unit"], "continent": "Континент"},
            template="plotly_white",
        )
        fig_ts.update_layout(hovermode="x unified", height=420, legend=dict(orientation="h", y=-0.22))

    # 3. Топ-10 стран (население >= 1 млн) по выбранной метрике
    agg_sql = "MAX" if meta["agg"] == "max" else "AVG"
    top_sql = f"""
        SELECT
            l.location,
            l.continent,
            ROUND({agg_sql}(t.{metric_col}), 2) AS metric_val
        FROM {table} t
        JOIN locations l ON t.iso_code = l.iso_code
        WHERE t.date BETWEEN ? AND ? {where_region}
          AND l.population >= 1000000
          AND t.{metric_col} IS NOT NULL
        GROUP BY l.iso_code, l.location, l.continent
        ORDER BY metric_val DESC
        LIMIT 10
    """
    top_df = query_sqlite(top_sql, params)
    if top_df.empty:
        fig_top = empty_figure("Нет стран для выбранного фильтра")
    else:
        top_df = top_df.sort_values("metric_val", ascending=True)
        fig_top = px.bar(
            top_df,
            x="metric_val",
            y="location",
            color="continent",
            orientation="h",
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
            x=ts_df["date"],
            y=ts_df["smoothed_pm"],
            mode="lines",
            name="7-дневное сглаживание (случаи / млн)",
            line=dict(color="#1d4ed8", width=2),
        )
    )
    if not waves_sub.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(waves_sub["peak_date"]),
                y=waves_sub["peak_cases_pm_smoothed"],
                mode="markers+text",
                name="Пики волн (find_peaks)",
                text=[f"W{n}" for n in waves_sub["wave_number"]],
                textposition="top center",
                marker=dict(color="#dc2626", size=10, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"{loc_name} ({iso_code}) — обнаружено волн: {len(waves_sub)}",
        xaxis_title="Дата",
        yaxis_title="Новые случаи на 1 млн (7d сглаж.)",
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True)),
        height=460,
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
    sql = f"""
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
    """
    df = query_sqlite(sql, params)
    if df.empty:
        return empty_figure("Нет данных для графика вакцинации и летальности")

    fig = px.scatter(
        df,
        x="full_vacc_pct",
        y="cfr_percent",
        size="population",
        color="continent",
        hover_name="location",
        hover_data={"deaths_pm": True, "gdp_per_capita": ":.0f", "population": ":,.0f"},
        title="Вакцинация (%) против Летальности CFR (%) по странам (население >= 1 млн)",
        labels={
            "full_vacc_pct": "Полная вакцинация (% населения)",
            "cfr_percent": "Летальность CFR (%)",
            "continent": "Континент",
        },
        template="plotly_white",
        size_max=42,
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
        go.Scatter(
            x=actual_df["date"],
            y=actual_df["actual"],
            mode="lines",
            name="Факт (actual)",
            line=dict(color="#0f172a", width=2.5),
        )
    )

    color_map = {
        "Linear Regression (AR lags)": "#dc2626",
        "Naive (baseline)": "#f59e0b",
        "Moving Average 7d": "#16a34a",
        "Seasonal Naive 7d (baseline)": "#7c3aed",
    }
    for model_name, grp in sub_p.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=grp["date"],
                y=grp["predicted"],
                mode="lines",
                name=model_name,
                line=dict(color=color_map.get(model_name, "#2563eb"), width=1.8),
            )
        )

    series_title = str(sub_m["series_name"].iloc[0]) if not sub_m.empty else series_code
    fig.update_layout(
        title=f"Факт против прогноза моделей (без data leakage): {series_title}",
        xaxis_title="Дата (тестовое окно)",
        yaxis_title="Новые случаи (7d сглаж.)",
        hovermode="x unified",
        template="plotly_white",
        height=420,
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
        page_size=15,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px 10px", "fontSize": "13px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
    )
    return info, table


def launch_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    print(f"\n[Dashboard] Запуск Dash-приложения: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    launch_dashboard()
