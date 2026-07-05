"""
Summary statistics for DAM_Data.csv

Computes mean, median, min, max, skewness, and kurtosis for each numeric
column in the dataset. By default, only the core (non-lagged) series are
reported -- EURPrices, WF, DF -- since the dataset also contains many
lagged versions of these columns (e.g. EURPrices-24, WF-1, DF-168).

Set CORE_ONLY = False below to compute statistics for every column instead.
"""

import pandas as pd

# ---- Configuration ----------------------------------------------------
DATA_PATH = "DAM_Data.csv"   # update this path if needed
CORE_ONLY = True             # True -> only EURPrices, WF, DF; False -> all columns
OUTPUT_PATH = "DAM_summary_statistics.csv"
# -------------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA_PATH)

    if CORE_ONLY:
        core_cols = [c for c in ["EURPrices", "WF", "DF"] if c in df.columns]
        data = df[core_cols]
    else:
        data = df.select_dtypes(include="number")

    summary = pd.DataFrame({
        "mean": data.mean(),
        "median": data.median(),
        "min": data.min(),
        "max": data.max(),
        "skewness": data.skew(),
        "kurtosis": data.kurtosis(),  # excess kurtosis (normal = 0)
    })

    print(summary.round(3))
    summary.round(3).to_csv(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
