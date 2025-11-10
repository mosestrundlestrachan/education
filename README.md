# Predictive Analysis of U.S. Educational Attainment: Forecasting Completion and Identifying Demographic Disparities

## Overview

This project provides a comprehensive analysis of U.S. educational attainment trends from 1920 to 2018 using historical data from the National Center for Education Statistics (NCES). The work is structured into five sequential steps, demonstrating core data science skills, including data cleaning, time series forecasting, and policy-focused data visualization.

The primary deliverables are:
1.  A **Time Series Forecast** of Bachelor's degree attainment (Total population) into the near future (2019-2023).
2.  A **Quantitative Equity Analysis** visualization comparing the attainment trajectory of the Hispanic demographic against the overall Total population.

---

## Data Source

The analysis uses one dataset:
* **Source:** [National Center for Education Statistics (NCES)](https://nces.ed.gov/)
* **Dataset:** *Percentage of persons 25 to 29 years old with selected levels of educational attainment, by race/ethnicity and sex: Selected years, 1920 through 2018.*
* **File:** `nces-ed-attainment.csv` (Must be included in the repository root.)

---

## Technical Stack & Dependencies

The project is executed entirely within Python, designed to run seamlessly in a Jupyter Notebook environment (such as VS Code's Notebook Editor).

| Tool | Purpose |
| :--- | :--- |
| **Python 3** | Core language for data processing and modeling. |
| **Pandas** | Data manipulation, MultiIndex handling, and imputation. |
| **Seaborn/Matplotlib** | Advanced data visualization and presentation. |
| **`statsmodels`** | Time series modeling (ARIMA). |

### Setup

To replicate the analysis, clone the repository and install the required libraries:

```bash
# Clone the repository
git clone [YOUR_REPO_URL]
cd [YOUR_REPO_NAME]

# Install required packages
pip install pandas seaborn matplotlib statsmodels
