-- name: q01_continent_summary
-- Техники: CTE + JOIN + GROUP BY + HAVING
WITH country_totals AS (
    SELECT
        e.iso_code,
        MAX(e.total_cases) AS max_cases,
        MAX(e.total_deaths) AS max_deaths
    FROM epidemiology e
    GROUP BY e.iso_code
)
SELECT
    l.continent,
    COUNT(DISTINCT l.iso_code) AS countries_count,
    ROUND(SUM(ct.max_cases), 0) AS total_cases,
    ROUND(SUM(ct.max_deaths), 0) AS total_deaths,
    ROUND(SUM(ct.max_deaths) * 100.0 / NULLIF(SUM(ct.max_cases), 0), 2) AS cfr_percent
FROM locations l
JOIN country_totals ct ON l.iso_code = ct.iso_code
GROUP BY l.continent
HAVING SUM(ct.max_cases) > 1000000
ORDER BY total_cases DESC;

-- name: q02_top15_mortality_per_million
-- Техники: JOIN + GROUP BY + HAVING
SELECT
    l.iso_code,
    l.location,
    l.continent,
    ROUND(l.population / 1000000.0, 2) AS population_mln,
    ROUND(MAX(e.total_deaths_per_million), 2) AS max_deaths_per_million,
    ROUND(MAX(e.total_cases_per_million), 2) AS max_cases_per_million
FROM locations l
JOIN epidemiology e ON l.iso_code = e.iso_code
WHERE l.population >= 1000000
GROUP BY l.iso_code, l.location, l.continent, l.population
HAVING MAX(e.total_deaths_per_million) IS NOT NULL
ORDER BY max_deaths_per_million DESC
LIMIT 15;

-- name: q03_above_avg_vaccination
-- Техники: JOIN + GROUP BY + HAVING + вложенный подзапрос
SELECT
    l.iso_code,
    l.location,
    l.continent,
    ROUND(MAX(v.people_fully_vaccinated_per_hundred), 2) AS max_full_vacc_pct
FROM locations l
JOIN vaccinations v ON l.iso_code = v.iso_code
GROUP BY l.iso_code, l.location, l.continent
HAVING MAX(v.people_fully_vaccinated_per_hundred) > (
    SELECT AVG(country_max)
    FROM (
        SELECT MAX(people_fully_vaccinated_per_hundred) AS country_max
        FROM vaccinations
        WHERE people_fully_vaccinated_per_hundred IS NOT NULL
        GROUP BY iso_code
    )
)
ORDER BY max_full_vacc_pct DESC;

-- name: q04_monthly_global_waves
-- Техники: GROUP BY + HAVING
SELECT
    STRFTIME('%Y-%m', date) AS year_month,
    ROUND(SUM(COALESCE(new_cases, 0)), 0) AS global_new_cases,
    ROUND(SUM(COALESCE(new_deaths, 0)), 0) AS global_new_deaths,
    COUNT(DISTINCT iso_code) AS reporting_countries
FROM epidemiology
GROUP BY year_month
HAVING SUM(COALESCE(new_cases, 0)) > 10000
ORDER BY year_month;

-- name: q05_rolling_7d_cases_window
-- Техники: JOIN + Оконная функция (AVG OVER ROWS BETWEEN)
SELECT
    l.location,
    e.iso_code,
    e.date,
    ROUND(e.new_cases, 1) AS new_cases,
    ROUND(
        AVG(e.new_cases) OVER (
            PARTITION BY e.iso_code
            ORDER BY e.date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_7d_cases_sql
FROM epidemiology e
JOIN locations l ON e.iso_code = l.iso_code
WHERE e.iso_code IN ('KAZ', 'USA', 'DEU', 'JPN', 'BRA')
  AND e.date BETWEEN '2020-03-01' AND '2023-12-31'
  AND e.new_cases IS NOT NULL
ORDER BY e.iso_code, e.date;

-- name: q06_top3_peaks_per_continent
-- Техники: CTE + JOIN + Оконная функция (RANK OVER PARTITION BY)
WITH country_peaks AS (
    SELECT
        l.continent,
        l.location,
        e.iso_code,
        MAX(e.new_cases_per_million) AS peak_cases_per_million
    FROM epidemiology e
    JOIN locations l ON e.iso_code = l.iso_code
    WHERE l.population >= 1000000
      AND e.new_cases_per_million IS NOT NULL
    GROUP BY l.continent, l.location, e.iso_code
),
ranked_peaks AS (
    SELECT
        continent,
        location,
        iso_code,
        ROUND(peak_cases_per_million, 2) AS peak_cases_per_million,
        RANK() OVER (
            PARTITION BY continent
            ORDER BY peak_cases_per_million DESC
        ) AS continent_rank
    FROM country_peaks
)
SELECT continent, continent_rank, location, iso_code, peak_cases_per_million
FROM ranked_peaks
WHERE continent_rank <= 3
ORDER BY continent, continent_rank;

-- name: q07_mortality_before_after_50pct_vacc
-- Техники: Multi-CTE + JOIN
WITH milestone_50 AS (
    SELECT
        iso_code,
        MIN(date) AS date_reached_50pct
    FROM vaccinations
    WHERE people_fully_vaccinated_per_hundred >= 50.0
    GROUP BY iso_code
),
mortality_split AS (
    SELECT
        m.iso_code,
        m.date_reached_50pct,
        AVG(CASE WHEN e.date < m.date_reached_50pct THEN e.new_deaths_per_million END) AS avg_deaths_pm_before,
        AVG(CASE WHEN e.date >= m.date_reached_50pct AND e.date <= '2023-12-31' THEN e.new_deaths_per_million END) AS avg_deaths_pm_after
    FROM milestone_50 m
    JOIN epidemiology e ON m.iso_code = e.iso_code
    WHERE e.new_deaths_per_million IS NOT NULL
      AND e.date >= '2020-03-01'
    GROUP BY m.iso_code, m.date_reached_50pct
)
SELECT
    l.continent,
    COUNT(*) AS countries_with_50pct_vacc,
    ROUND(AVG(ms.avg_deaths_pm_before), 3) AS mean_deaths_pm_before_50pct,
    ROUND(AVG(ms.avg_deaths_pm_after), 3) AS mean_deaths_pm_after_50pct,
    ROUND(
        (AVG(ms.avg_deaths_pm_after) - AVG(ms.avg_deaths_pm_before)) * 100.0
        / NULLIF(AVG(ms.avg_deaths_pm_before), 0),
        1
    ) AS pct_change
FROM mortality_split ms
JOIN locations l ON ms.iso_code = l.iso_code
WHERE ms.avg_deaths_pm_before IS NOT NULL AND ms.avg_deaths_pm_after IS NOT NULL
GROUP BY l.continent
ORDER BY pct_change ASC;

-- name: q08_stringency_and_icu_by_year
-- Техники: Тройной JOIN + GROUP BY + HAVING
SELECT
    l.continent,
    STRFTIME('%Y', e.date) AS year,
    ROUND(AVG(h.stringency_index), 2) AS avg_stringency_index,
    ROUND(AVG(h.icu_patients), 1) AS avg_icu_patients,
    ROUND(AVG(e.new_deaths_per_million), 3) AS avg_new_deaths_per_million
FROM epidemiology e
JOIN locations l ON e.iso_code = l.iso_code
JOIN hospital_policy h ON e.iso_code = h.iso_code AND e.date = h.date
WHERE STRFTIME('%Y', e.date) BETWEEN '2020' AND '2023'
GROUP BY l.continent, year
HAVING AVG(h.stringency_index) IS NOT NULL
ORDER BY l.continent, year;

-- name: q09_kazakhstan_yearly_yoy
-- Техники: CTE + LEFT JOIN + Оконная функция (LAG)
WITH kaz_yearly AS (
    SELECT
        STRFTIME('%Y', e.date) AS year,
        ROUND(SUM(COALESCE(e.new_cases, 0)), 0) AS yearly_cases,
        ROUND(SUM(COALESCE(e.new_deaths, 0)), 0) AS yearly_deaths,
        ROUND(MAX(v.people_fully_vaccinated_per_hundred), 2) AS max_vacc_pct
    FROM epidemiology e
    LEFT JOIN vaccinations v ON e.iso_code = v.iso_code AND e.date = v.date
    WHERE e.iso_code = 'KAZ'
    GROUP BY year
)
SELECT
    year,
    yearly_cases,
    LAG(yearly_cases) OVER (ORDER BY year) AS prev_year_cases,
    ROUND(
        (yearly_cases - LAG(yearly_cases) OVER (ORDER BY year)) * 100.0
        / NULLIF(LAG(yearly_cases) OVER (ORDER BY year), 0),
        1
    ) AS cases_yoy_pct,
    yearly_deaths,
    max_vacc_pct
FROM kaz_yearly
ORDER BY year;
