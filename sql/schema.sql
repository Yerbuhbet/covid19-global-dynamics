PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS hospital_policy;
DROP TABLE IF EXISTS vaccinations;
DROP TABLE IF EXISTS epidemiology;
DROP TABLE IF EXISTS locations;

CREATE TABLE locations (
    iso_code TEXT PRIMARY KEY,
    continent TEXT NOT NULL,
    location TEXT NOT NULL,
    population REAL,
    population_density REAL,
    median_age REAL,
    gdp_per_capita REAL,
    life_expectancy REAL,
    human_development_index REAL
);

CREATE TABLE epidemiology (
    iso_code TEXT NOT NULL,
    date TEXT NOT NULL,
    total_cases REAL,
    new_cases REAL,
    new_cases_smoothed REAL,
    total_deaths REAL,
    new_deaths REAL,
    new_deaths_smoothed REAL,
    total_cases_per_million REAL,
    total_deaths_per_million REAL,
    new_cases_per_million REAL,
    new_deaths_per_million REAL,
    reproduction_rate REAL,
    PRIMARY KEY (iso_code, date),
    FOREIGN KEY (iso_code) REFERENCES locations(iso_code)
);

CREATE TABLE vaccinations (
    iso_code TEXT NOT NULL,
    date TEXT NOT NULL,
    total_vaccinations REAL,
    people_vaccinated REAL,
    people_fully_vaccinated REAL,
    total_boosters REAL,
    new_vaccinations REAL,
    new_vaccinations_smoothed REAL,
    people_fully_vaccinated_per_hundred REAL,
    PRIMARY KEY (iso_code, date),
    FOREIGN KEY (iso_code) REFERENCES locations(iso_code)
);

CREATE TABLE hospital_policy (
    iso_code TEXT NOT NULL,
    date TEXT NOT NULL,
    icu_patients REAL,
    hosp_patients REAL,
    stringency_index REAL,
    PRIMARY KEY (iso_code, date),
    FOREIGN KEY (iso_code) REFERENCES locations(iso_code)
);

CREATE INDEX idx_epi_date ON epidemiology(date);
CREATE INDEX idx_vacc_date ON vaccinations(date);
CREATE INDEX idx_hosp_date ON hospital_policy(date);
CREATE INDEX idx_loc_continent ON locations(continent);
