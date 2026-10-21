import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

OUTCOMES_FILE = "C:\\Users\\arham\\OneDrive - BIRLA INSTITUTE OF TECHNOLOGY and SCIENCE\\Documents\\wavelet_market_analysis\\outcomes\\nifty50_future_outcomes.csv"

WAVELET_FILES = {
    "Haar": "percentiles/haar_swt_percentiles.csv",
    "DB4": "percentiles/db4_swt_percentiles.csv",
    "Sym4": "percentiles/sym4_swt_percentiles.csv",
    "Coif1": "percentiles/coif1_swt_percentiles.csv"
}

OUTPUT_DIR = "si_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD FUTURE OUTCOMES
# ============================================================

outcomes = pd.read_csv(OUTCOMES_FILE)

outcomes["Date"] = pd.to_datetime(outcomes["Date"])

print("=" * 70)
print("FUTURE OUTCOMES")
print("=" * 70)
print("Rows:", len(outcomes))
print("Date range:", outcomes["Date"].min(), "to", outcomes["Date"].max())


# ============================================================
# STRESS DEFINITIONS
# ============================================================

# Return stress
outcomes["Stress_Return_5D"] = outcomes["Future_Return_5D"] < -0.03
outcomes["Stress_Return_10D"] = outcomes["Future_Return_10D"] < -0.03
outcomes["Stress_Return_20D"] = outcomes["Future_Return_20D"] < -0.03

# More severe return stress
outcomes["Stress_Return_5D_Severe"] = outcomes["Future_Return_5D"] < -0.05

# Maximum drawdown stress
outcomes["Stress_MDD_5D"] = outcomes["Future_MDD_5D"] < -0.03
outcomes["Stress_MDD_10D"] = outcomes["Future_MDD_10D"] < -0.03
outcomes["Stress_MDD_20D"] = outcomes["Future_MDD_20D"] < -0.03


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_si(wavelet_name, percentile_file):

    print("\n")
    print("=" * 70)
    print(f"{wavelet_name.upper()} — SI-ONLY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load percentile data
    # --------------------------------------------------------

    wavelet = pd.read_csv(percentile_file)

    wavelet["Date"] = pd.to_datetime(wavelet["Date"])

    # --------------------------------------------------------
    # Calculate SI
    # --------------------------------------------------------

    wavelet["HF"] = (
        wavelet["Q_D1"] +
        wavelet["Q_D2"]
    ) / 2

    wavelet["LF"] = (
        wavelet["Q_D3"] +
        wavelet["Q_A3"]
    ) / 2

    wavelet["SI"] = wavelet["HF"] - wavelet["LF"]

    # --------------------------------------------------------
    # Merge with future outcomes
    # --------------------------------------------------------

    data = pd.merge(
        wavelet,
        outcomes,
        on="Date",
        how="inner"
    )

    # Remove rows where SI is unavailable
    data = data.dropna(subset=["SI"]).copy()

    print("Valid observations:", len(data))

    # ========================================================
    # BASIC SI STATISTICS
    # ========================================================

    print("\n--- SI DISTRIBUTION ---")

    print("Mean   :", data["SI"].mean())
    print("Median :", data["SI"].median())
    print("Std    :", data["SI"].std())
    print("Min    :", data["SI"].min())
    print("Max    :", data["SI"].max())

    print("\nSI percentiles:")

    print(
        data["SI"].quantile(
            [0.01, 0.05, 0.10, 0.25,
             0.50,
             0.75, 0.90, 0.95, 0.99]
        )
    )

    # ========================================================
    # SI QUARTILES
    # ========================================================

    data["SI_Quartile"] = pd.qcut(
        data["SI"],
        q=4,
        labels=["Q1_Low", "Q2", "Q3", "Q4_High"]
    )

    # ========================================================
    # BASELINE PROBABILITIES
    # ========================================================

    stress_columns = [
        "Stress_Return_5D",
        "Stress_Return_5D_Severe",
        "Stress_Return_10D",
        "Stress_Return_20D",
        "Stress_MDD_5D",
        "Stress_MDD_10D",
        "Stress_MDD_20D"
    ]

    print("\n--- UNCONDITIONAL STRESS PROBABILITIES ---")

    baseline = {}

    for col in stress_columns:
        p = data[col].mean()
        baseline[col] = p

        print(
            f"{col:28s}: "
            f"{p:.4%}"
        )

    # ========================================================
    # QUARTILE ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SI QUARTILE ANALYSIS")
    print("=" * 70)

    results = []

    for quartile in ["Q1_Low", "Q2", "Q3", "Q4_High"]:

        subset = data[
            data["SI_Quartile"] == quartile
        ]

        print(f"\n{quartile}")
        print("-" * 50)

        print(
            "SI range:",
            round(subset["SI"].min(), 3),
            "to",
            round(subset["SI"].max(), 3)
        )

        print("Observations:", len(subset))

        row = {
            "Wavelet": wavelet_name,
            "SI_Quartile": quartile,
            "N": len(subset),
            "SI_Mean": subset["SI"].mean(),
            "SI_Median": subset["SI"].median()
        }

        for col in stress_columns:

            probability = subset[col].mean()

            rr = (
                probability / baseline[col]
                if baseline[col] > 0
                else np.nan
            )

            row[f"{col}_Probability"] = probability
            row[f"{col}_RelativeRisk"] = rr

            print(
                f"{col:28s}: "
                f"{probability:.4%} "
                f"(RR = {rr:.3f})"
            )

        results.append(row)

    quartile_results = pd.DataFrame(results)

    # ========================================================
    # EXTREME SI ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EXTREME SI ANALYSIS")
    print("=" * 70)

    thresholds = [
        ("SI <= -10", data["SI"] <= -10),
        ("SI <= -15", data["SI"] <= -15),
        ("SI <= -20", data["SI"] <= -20),
        ("SI >= 10", data["SI"] >= 10),
        ("SI >= 15", data["SI"] >= 15),
        ("SI >= 20", data["SI"] >= 20)
    ]

    extreme_results = []

    for label, mask in thresholds:

        subset = data[mask]

        print(f"\n{label}")
        print("-" * 50)

        print("Observations:", len(subset))

        row = {
            "Wavelet": wavelet_name,
            "Threshold": label,
            "N": len(subset)
        }

        if len(subset) == 0:
            extreme_results.append(row)
            continue

        row["SI_Mean"] = subset["SI"].mean()

        for col in stress_columns:

            probability = subset[col].mean()

            rr = (
                probability / baseline[col]
                if baseline[col] > 0
                else np.nan
            )

            row[f"{col}_Probability"] = probability
            row[f"{col}_RelativeRisk"] = rr

            print(
                f"{col:28s}: "
                f"{probability:.4%} "
                f"(RR = {rr:.3f})"
            )

        extreme_results.append(row)

    extreme_results = pd.DataFrame(extreme_results)

    # ========================================================
    # SI BIN ANALYSIS — 10 QUANTILE BINS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SI DECILE ANALYSIS")
    print("=" * 70)

    data["SI_Decile"] = pd.qcut(
        data["SI"],
        q=10,
        labels=False
    ) + 1

    decile_results = []

    for decile in range(1, 11):

        subset = data[
            data["SI_Decile"] == decile
        ]

        row = {
            "Wavelet": wavelet_name,
            "SI_Decile": decile,
            "N": len(subset),
            "SI_Mean": subset["SI"].mean(),
            "SI_Median": subset["SI"].median()
        }

        print(
            f"\nDecile {decile}: "
            f"SI mean = {subset['SI'].mean():.3f}"
        )

        for col in stress_columns:

            probability = subset[col].mean()

            rr = (
                probability / baseline[col]
                if baseline[col] > 0
                else np.nan
            )

            row[f"{col}_Probability"] = probability
            row[f"{col}_RelativeRisk"] = rr

            print(
                f"  {col:26s}: "
                f"{probability:.4%} "
                f"(RR={rr:.3f})"
            )

        decile_results.append(row)

    decile_results = pd.DataFrame(decile_results)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    quartile_results.to_csv(
        f"{OUTPUT_DIR}/{wavelet_name.lower()}_si_quartiles.csv",
        index=False
    )

    extreme_results.to_csv(
        f"{OUTPUT_DIR}/{wavelet_name.lower()}_si_extremes.csv",
        index=False
    )

    decile_results.to_csv(
        f"{OUTPUT_DIR}/{wavelet_name.lower()}_si_deciles.csv",
        index=False
    )

    # Save merged data too
    data.to_csv(
        f"{OUTPUT_DIR}/{wavelet_name.lower()}_si_merged.csv",
        index=False
    )

    return data, quartile_results, extreme_results, decile_results


# ============================================================
# RUN FOR ALL WAVELETS
# ============================================================

all_quartiles = []
all_extremes = []
all_deciles = []

for wavelet_name, filename in WAVELET_FILES.items():

    data, q, e, d = analyze_si(
        wavelet_name,
        filename
    )

    all_quartiles.append(q)
    all_extremes.append(e)
    all_deciles.append(d)


# ============================================================
# COMBINED RESULTS
# ============================================================

combined_quartiles = pd.concat(
    all_quartiles,
    ignore_index=True
)

combined_extremes = pd.concat(
    all_extremes,
    ignore_index=True
)

combined_deciles = pd.concat(
    all_deciles,
    ignore_index=True
)

combined_quartiles.to_csv(
    f"{OUTPUT_DIR}/ALL_wavelets_si_quartiles.csv",
    index=False
)

combined_extremes.to_csv(
    f"{OUTPUT_DIR}/ALL_wavelets_si_extremes.csv",
    index=False
)

combined_deciles.to_csv(
    f"{OUTPUT_DIR}/ALL_wavelets_si_deciles.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SI ANALYSIS COMPLETE")
print("=" * 70)

print("\nFiles saved in:", OUTPUT_DIR)

print("""
Main outputs:

1. *_si_quartiles.csv
   -> Q1 vs Q2 vs Q3 vs Q4

2. *_si_extremes.csv
   -> SI <= -10/-15/-20
   -> SI >= 10/15/20

3. *_si_deciles.csv
   -> Full SI decile relationship

4. *_si_merged.csv
   -> Complete SI + future outcome dataset
""")