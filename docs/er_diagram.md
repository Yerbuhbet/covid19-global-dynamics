# ER-диаграмма базы данных SQLite (`covid.db`)

Схема нормализована в 3NF: статические атрибуты стран вынесены в справочник `locations`, а ежедневные наблюдения разделены по доменам на 3 таблицы фактов с составным первичным ключом `(iso_code, date)` и внешним ключом на `locations(iso_code)`.

```mermaid
erDiagram
    locations ||--o{ epidemiology : "has daily cases/deaths"
    locations ||--o{ vaccinations : "has vaccination records"
    locations ||--o{ hospital_policy : "has hospital & policy metrics"

    locations {
        TEXT iso_code PK
        TEXT continent
        TEXT location
        REAL population
        REAL population_density
        REAL median_age
        REAL gdp_per_capita
        REAL life_expectancy
        REAL human_development_index
    }

    epidemiology {
        TEXT iso_code PK, FK
        TEXT date PK
        REAL total_cases
        REAL new_cases
        REAL new_cases_smoothed
        REAL total_deaths
        REAL new_deaths
        REAL new_deaths_smoothed
        REAL total_cases_per_million
        REAL total_deaths_per_million
        REAL new_cases_per_million
        REAL new_deaths_per_million
        REAL reproduction_rate
    }

    vaccinations {
        TEXT iso_code PK, FK
        TEXT date PK
        REAL total_vaccinations
        REAL people_vaccinated
        REAL people_fully_vaccinated
        REAL total_boosters
        REAL new_vaccinations
        REAL new_vaccinations_smoothed
        REAL people_fully_vaccinated_per_hundred
    }

    hospital_policy {
        TEXT iso_code PK, FK
        TEXT date PK
        REAL icu_patients
        REAL hosp_patients
        REAL stringency_index
    }
```
