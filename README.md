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
- Exports results to Excel

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

Output: `figures/` (6 PNGs) and `results.xlsx`.
