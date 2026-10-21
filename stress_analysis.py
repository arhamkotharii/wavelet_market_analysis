import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "outcomes/nifty50_future_outcomes.csv"
OUTPUT_DIR = "stress"

HORIZONS = [5, 10, 20]
THRESHOLDS = [-0.03, -0.05, -0.07]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# ANALYZE RETURN-BASED STRESS EVENTS
# ============================================================

def analyze_return_stress(df):

    results = []

    for horizon in HORIZONS:

        column = f"Future_Return_{horizon}D"

        valid = df[column].dropna()

        print("\n" + "-" * 70)
        print(f"{horizon}-DAY FUTURE RETURN")
        print("-" * 70)

        print(f"Valid observations : {len(valid)}")
        print(f"Mean               : {valid.mean():.6f}")
        print(f"Median             : {valid.median():.6f}")
        print(f"Std                 : {valid.std():.6f}")
        print(f"Minimum            : {valid.min():.6f}")
        print(f"Maximum            : {valid.max():.6f}")

        for threshold in THRESHOLDS:

            stress = valid < threshold

            count = stress.sum()
            percentage = stress.mean() * 100

            results.append({
                "Horizon": horizon,
                "Threshold": threshold,
                "Stress_Events": count,
                "Valid_Observations": len(valid),
                "Stress_Rate_Percent": percentage
            })

            print(
                f"Stress < {threshold:.1%}: "
                f"{count:4d} events "
                f"({percentage:.2f}%)"
            )

    return pd.DataFrame(results)


# ============================================================
# ANALYZE FUTURE VOLATILITY
# ============================================================

def analyze_future_volatility(df):

    results = []

    for horizon in HORIZONS:

        column = f"Future_RV_{horizon}D"

        valid = df[column].dropna()

        results.append({
            "Horizon": horizon,
            "Mean": valid.mean(),
            "Median": valid.median(),
            "Std": valid.std(),
            "P90": valid.quantile(0.90),
            "P95": valid.quantile(0.95),
            "P97_5": valid.quantile(0.975),
            "P99": valid.quantile(0.99),
            "Maximum": valid.max()
        })

    return pd.DataFrame(results)


# ============================================================
# ANALYZE FUTURE MAXIMUM DRAWDOWN
# ============================================================

def analyze_future_drawdown(df):

    results = []

    for horizon in HORIZONS:

        column = f"Future_MDD_{horizon}D"

        valid = df[column].dropna()

        results.append({
            "Horizon": horizon,
            "Mean": valid.mean(),
            "Median": valid.median(),
            "Std": valid.std(),
            "P10": valid.quantile(0.10),
            "P5": valid.quantile(0.05),
            "P2_5": valid.quantile(0.025),
            "P1": valid.quantile(0.01),
            "Minimum": valid.min()
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
    print("INDEPENDENT FUTURE STRESS ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(
        f"Input observations : {len(df)}"
    )

    # --------------------------------------------------------
    # RETURN-BASED STRESS
    # --------------------------------------------------------

    stress_results = analyze_return_stress(df)

    stress_path = os.path.join(
        OUTPUT_DIR,
        "return_stress_frequency.csv"
    )

    stress_results.to_csv(
        stress_path,
        index=False
    )

    # --------------------------------------------------------
    # FUTURE VOLATILITY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUTURE REALIZED VOLATILITY")
    print("=" * 70)

    volatility_results = analyze_future_volatility(df)

    print(
        volatility_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    volatility_path = os.path.join(
        OUTPUT_DIR,
        "future_volatility_distribution.csv"
    )

    volatility_results.to_csv(
        volatility_path,
        index=False
    )

    # --------------------------------------------------------
    # FUTURE MAXIMUM DRAWDOWN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUTURE MAXIMUM DRAWDOWN")
    print("=" * 70)

    drawdown_results = analyze_future_drawdown(df)

    print(
        drawdown_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    drawdown_path = os.path.join(
        OUTPUT_DIR,
        "future_drawdown_distribution.csv"
    )

    drawdown_results.to_csv(
        drawdown_path,
        index=False
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRESS ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {stress_path}"
    )

    print(
        f"Saved: {volatility_path}"
    )

    print(
        f"Saved: {drawdown_path}"
    )


if __name__ == "__main__":
    main()