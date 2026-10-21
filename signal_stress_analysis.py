import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OUTCOME_FILE = "outcomes/nifty50_future_outcomes.csv"
PERCENTILE_DIR = "percentiles"
OUTPUT_DIR = "stress"

WAVELETS = ["haar", "db4", "sym4", "coif1"]

MAS_BINS = [
    ("<75", -np.inf, 75.0),
    ("75-90", 75.0, 90.0),
    ("90-95", 90.0, 95.0),
    ("95-97.5", 95.0, 97.5),
    (">=97.5", 97.5, np.inf)
]

STRESS_COLUMN = "Future_Return_5D"
STRESS_THRESHOLD = -0.03


# ============================================================
# LOAD OUTCOMES
# ============================================================

def load_outcomes():

    df = pd.read_csv(
        OUTCOME_FILE,
        parse_dates=["Date"]
    )

    return df[
        [
            "Date",
            "Future_Return_5D",
            "Future_RV_5D",
            "Future_MDD_5D",
            "Future_Return_20D",
            "Future_RV_20D",
            "Future_MDD_20D"
        ]
    ]

# ============================================================
# LOAD PERCENTILE DATA
# ============================================================

def load_percentiles(wavelet):

    filename = f"{wavelet}_swt_percentiles.csv"

    path = os.path.join(
        PERCENTILE_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing percentile file: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"]
    )

    return df[
        [
            "Date",
            "Q_D1",
            "Q_D2",
            "Q_D3",
            "Q_A3",
            "MAS"
        ]
    ]


# ============================================================
# MERGE CURRENT SIGNAL WITH FUTURE OUTCOMES
# ============================================================

def build_analysis_dataset(wavelet):

    percentiles = load_percentiles(wavelet)

    outcomes = load_outcomes()

    df = pd.merge(
        percentiles,
        outcomes,
        on="Date",
        how="inner"
    )

    # Only observations where both the signal and
    # future 20-day outcome are available.
    df = df.dropna(
        subset=[
            "MAS",
            STRESS_COLUMN
        ]
    ).copy()

    return df


# ============================================================
# DEFINE STRESS
# ============================================================

def add_stress_indicator(df):

    df["Stress_20D_5pct"] = (
        df[STRESS_COLUMN] < STRESS_THRESHOLD
    ).astype(int)

    return df


# ============================================================
# MAS BIN
# ============================================================

def assign_mas_bin(mas):

    for label, lower, upper in MAS_BINS:

        if lower <= mas < upper:
            return label

    return np.nan


# ============================================================
# CONDITIONAL STRESS ANALYSIS
# ============================================================

def calculate_mas_stress_rates(df, wavelet):

    baseline_rate = (
        df["Stress_20D_5pct"].mean()
        * 100
    )

    rows = []

    for label, lower, upper in MAS_BINS:

        mask = (
            (df["MAS"] >= lower)
            & (df["MAS"] < upper)
        )

        subset = df.loc[mask]

        count = len(subset)

        if count == 0:
            continue

        stress_events = (
            subset["Stress_20D_5pct"].sum()
        )

        stress_rate = (
            stress_events / count * 100
        )

        relative_risk = (
            stress_rate / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        rows.append({
            "Wavelet": wavelet,
            "MAS_Bin": label,
            "Observations": count,
            "Stress_Events": stress_events,
            "Stress_Rate_Percent": stress_rate,
            "Baseline_Stress_Rate_Percent":
                baseline_rate,
            "Relative_Risk": relative_risk
        })

    return pd.DataFrame(rows)


# ============================================================
# SCALE-SPECIFIC STRESS ANALYSIS
# ============================================================

def calculate_scale_stress_rates(df, wavelet):

    baseline_rate = (
        df["Stress_20D_5pct"].mean()
        * 100
    )

    results = []

    percentile_columns = [
        "Q_D1",
        "Q_D2",
        "Q_D3",
        "Q_A3"
    ]

    for feature in percentile_columns:

        high_signal = df[feature] >= 95

        subset = df.loc[high_signal]

        count = len(subset)

        if count == 0:
            continue

        events = (
            subset["Stress_20D_5pct"].sum()
        )

        rate = events / count * 100

        relative_risk = (
            rate / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        results.append({
            "Wavelet": wavelet,
            "Feature": feature,
            "Threshold": 95,
            "Observations": count,
            "Stress_Events": events,
            "Stress_Rate_Percent": rate,
            "Baseline_Stress_Rate_Percent":
                baseline_rate,
            "Relative_Risk": relative_risk
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("WAVELET SIGNAL → FUTURE MARKET STRESS ANALYSIS")
    print("=" * 70)

    all_mas_results = []
    all_scale_results = []

    for wavelet in WAVELETS:

        print("\n" + "=" * 70)
        print(f"WAVELET: {wavelet.upper()}")
        print("=" * 70)

        df = build_analysis_dataset(wavelet)

        df = add_stress_indicator(df)

        baseline_rate = (
            df["Stress_20D_5pct"].mean()
            * 100
        )

        print(
            f"Valid paired observations : {len(df)}"
        )

        print(
            f"Baseline 20D stress rate  : "
            f"{baseline_rate:.2f}%"
        )

        # ----------------------------------------------------
        # MAS BINS
        # ----------------------------------------------------

        mas_results = calculate_mas_stress_rates(
            df,
            wavelet
        )

        all_mas_results.append(
            mas_results
        )

        print("\nMAS → Future Stress")

        print(
            mas_results[
                [
                    "MAS_Bin",
                    "Observations",
                    "Stress_Events",
                    "Stress_Rate_Percent",
                    "Relative_Risk"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

        # ----------------------------------------------------
        # SCALE-SPECIFIC SIGNALS
        # ----------------------------------------------------

        scale_results = calculate_scale_stress_rates(
            df,
            wavelet
        )

        all_scale_results.append(
            scale_results
        )

        print("\nScale-specific signals ≥95th percentile")

        print(
            scale_results[
                [
                    "Feature",
                    "Observations",
                    "Stress_Events",
                    "Stress_Rate_Percent",
                    "Relative_Risk"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    mas_final = pd.concat(
        all_mas_results,
        ignore_index=True
    )

    scale_final = pd.concat(
        all_scale_results,
        ignore_index=True
    )

    mas_path = os.path.join(
        OUTPUT_DIR,
        "mas_stress_analysis.csv"
    )

    scale_path = os.path.join(
        OUTPUT_DIR,
        "scale_stress_analysis.csv"
    )

    mas_final.to_csv(
        mas_path,
        index=False
    )

    scale_final.to_csv(
        scale_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {mas_path}"
    )

    print(
        f"Saved: {scale_path}"
    )


if __name__ == "__main__":
    main()