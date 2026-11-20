from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SQL_DIR = ROOT_DIR / "sql"
OUTPUTS_DIR = ROOT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RAW_CSV_PATH = RAW_DIR / "compact.csv"
CLEAN_COUNTRIES_PARQUET = PROCESSED_DIR / "covid_countries_clean.parquet"
CLEAN_AGGREGATES_PARQUET = PROCESSED_DIR / "covid_aggregates.parquet"
DB_PATH = PROCESSED_DIR / "covid.db"

EDA_AUDIT_PATH = METRICS_DIR / "eda_audit.json"
OUTLIERS_SUMMARY_PATH = METRICS_DIR / "outliers_iqr.csv"
STORAGE_BENCHMARK_PATH = METRICS_DIR / "storage_benchmark.json"

CHUNK_SIZE = 100_000
COLUMN_RENAME_MAP = {
    "code": "iso_code",
    "country": "location",
}

IQR_TARGET_COLUMNS = [
    "new_cases",
    "new_deaths",
    "new_cases_per_million",
    "new_deaths_per_million",
    "people_fully_vaccinated_per_hundred",
    "gdp_per_capita",
]

NON_NEGATIVE_COLUMNS = [
    "new_cases",
    "new_deaths",
    "new_cases_smoothed",
    "new_deaths_smoothed",
    "total_cases",
    "total_deaths",
]
