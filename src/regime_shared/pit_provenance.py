from __future__ import annotations

SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)
SOURCE_NAME = "fja05680/sp500"
# Downstream quality reports grep this exact token; change it only with a
# matching migration for stored bias-warning artifacts.
BIAS_WARNING = "survivorship_biased_constituent_universe"
