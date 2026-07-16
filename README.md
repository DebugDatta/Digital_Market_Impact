# Digital Market Impact

A comparative analysis of market volatility between **algorithm-driven stocks** (Nykaa, ETERNAL, Swiggy, Paytm) and **traditional FMCG stocks** (HUL, Britannia, Dabur, Marico, ITC) on the NSE.

## Features

- Downloads price data via `yfinance`
- Computes log returns, rolling/annualized volatility
- Statistical tests: ADF stationarity, ARCH-LM, Jarque-Bera normality
- GARCH(1,1) volatility modeling
- Structural break detection (PELT algorithm via `ruptures`)
- Group comparison: Welch's t-test & Levene's test
- Generates 6 publication-quality figures
- Exports results to Excel (4 sheets)

## Requirements

```
numpy==1.26.4
pandas==2.1.4
yfinance==1.4.1
scipy==1.17.0
statsmodels==0.14.6
arch==7.0.0
ruptures==1.1.10
matplotlib==3.10.7
openpyxl==3.1.5
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Output

### Figures (`figures/`)

| # | File | Description |
|---|------|-------------|
| 1 | `fig1_price_comparison.png` | Normalized price trends (base=100) for all stocks |
| 2 | `fig2_rolling_volatility.png` | Rolling 21-day annualized volatility (%) |
| 3 | `fig3_boxplot_returns.png` | Daily return distribution by group (algorithm vs traditional) |
| 4 | `fig4_correlation_heatmap.png` | Pearson correlation matrix of stock returns |
| 5 | `fig5_garch_conditional_volatility.png` | GARCH(1,1) conditional volatility for ETERNAL, PAYTM, HINDUNILVR, BRITANNIA |
| 6 | `fig6_structural_breaks.png` | Structural break points (PELT) for a representative stock |

### Excel (`results.xlsx`)

| Sheet | Contents |
|-------|----------|
| **Descriptive Statistics** | Mean return (%), Std Dev (%), Annualized Vol (%), Skewness, Kurtosis, Min/Max (%) per stock |
| **Statistical Tests** | ADF test (stationarity), ARCH-LM test (heteroskedasticity), Jarque-Bera test (normality) — statistic & p-value per stock |
| **GARCH Results** | GARCH(1,1) parameters — Omega, Alpha, Beta, Persistence (α+β), AIC, BIC, Log-Likelihood per stock |
| **Structural Breaks** | Break dates detected via PELT algorithm for each stock |
