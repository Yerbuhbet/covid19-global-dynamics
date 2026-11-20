# COVID-19 Global Dynamics — Analytics, Wave Detection & Forecasting

Итоговый мини-проект по дисциплине *Programming for Data Analysis and Database* (Cluster 1: Exploratory Analytics, Time Series & Dashboards).

- **Группа:** AAI-2606M (Astana IT University)
- **Автор / Команда:** Yermukhamet Akbet *(при командной сдаче указать: Alen Smailov, Yermukhamet Akbet, Sayan Aukatov)*
- **Источник данных:** Our World in Data (OWID) COVID-19 ETL Catalog — `compact.csv` (629 868 строк, 61 колонка, 176.74 МБ)
- **Стек:** Python 3.x, Pandas, PyArrow (Parquet), SQLite, SciPy, Statsmodels, Scikit-learn, Matplotlib, Seaborn, Plotly, Dash

## Структура проекта

```text
covid19-akbet/
├── data/
│   ├── raw/                     # Исходный compact.csv (скачивается по ссылке, не коммитится)
│   └── processed/               # Очищенный Parquet и нормализованная БД covid.db
├── sql/
│   ├── schema.sql               # DDL-схема (4 таблицы в 3NF + индексы)
│   └── queries.sql              # 9 аналитических SQL-запросов (JOIN, GROUP BY/HAVING, CTE, Window)
├── src/
│   ├── config.py                # Конфигурация путей и параметров
│   ├── data_loader.py           # Чтение chunksize, очистка, оптимизация RAM, IQR, CSV vs Parquet
│   ├── database.py              # Сборка SQLite и выгрузка 9 SQL-запросов в Pandas
│   ├── analytics.py             # Boxplot/skewness, resample, декомпозиция, корреляции, детекция волн
│   ├── forecasting.py           # Leakage-free сравнение моделей (Naive, Seasonal Naive, MA, AR LinReg)
│   └── main.py                  # Консольное меню управления пайплайном
├── outputs/
│   ├── figures/                 # 5 статических PNG-графиков + 1 интерактивный Plotly HTML
│   ├── metrics/                 # JSON/CSV таблицы замеров, волн и метрик прогноза
│   └── sql/                     # Выгрузки всех 9 SQL-запросов в CSV
├── docs/
│   ├── er_diagram.md            # ER-диаграмма БД (Mermaid)
│   └── report.md                # Письменный отчёт по проекту
├── app.py                       # Интерактивный веб-дашборд (Dash + Plotly)
├── requirements.txt             # Зафиксированные версии зависимостей
└── README.md
```

## Инструкция по запуску с нуля

### 1. Клонирование и установка окружения
```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd covid19-akbet
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Скачивание датасета Our World in Data
```bash
mkdir -p data/raw
curl -L -o data/raw/compact.csv https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv
```

### 3. Запуск пайплайна (два способа)

**Способ А — пошагово (или через меню `python -m src.main`):**
```bash
python -m src.data_loader        # Шаг 1: chunksize, очистка, Parquet, IQR, бенчмарк CSV vs Parquet
python -m src.database           # Шаг 2: SQLite (4 таблицы) + 9 SQL-запросов -> outputs/sql/
python -m src.analytics          # Шаг 3: EDA, декомпозиция, heatmap, детекция волн, Plotly HTML
python -m src.forecasting        # Шаг 4: Прогнозирование без data leakage (MAE/RMSE/nMAE)
```

**Способ Б — интерактивное консольное меню:**
```bash
python -m src.main
```

### 4. Запуск интерактивного дашборда Dash
```bash
python app.py
```
Откройте в браузере: `http://127.0.0.1:8050`

## Основные результаты
- **Оптимизация хранения (CSV vs Parquet):** чтение всех 61 колонок ускорено в **15.27x** (`0.546s -> 0.036s`), чтение 10 рабочих колонок — в **21.75x** (`0.345s -> 0.016s`), размер файла сжат в **9.91x** (`176.74 -> 17.84 МБ`), RAM снижена на **52.55%**.
- **База данных SQLite:** нормализованная схема из 4 таблиц (`237` стран, `577 391` эпид. записей) и **9 SQL-запросов**; на всех 6 континентах смертность на 1 млн снизилась на **44.8%–86.4%** после прохождения порога 50% полной вакцинации.
- **Детекция волн:** обнаружено **789 волн в 217 странах** (включая 5 волн в Казахстане с пиком Омикрона 23.01.2022 — `693.5` случаев/млн).
- **Прогнозирование (без утечки данных):** авторегрессионная `Linear Regression (AR lags)` показала лучший средний `nMAE = 3.15%` против `Naive (4.10%)`, `Moving Average 7d (11.90%)` и `Seasonal Naive 7d (20.48%)` (по Казахстану MAE снижен с `44.77` до `17.36`).
