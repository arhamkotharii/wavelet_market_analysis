import os
import numpy as np
import pandas as pd

PERCENTILE_DIR = "percentiles"
OUTCOME_FILE = "outcomes/nifty50_future_outcomes.csv"
OUTPUT_DIR = "stress"

WAVELETS = ["haar", "db4", "sym4", "coif1"]

# Future 5-day maximum drawdown threshold
STRESS_THRESHOLD = -0.03


def load_data(wavelet):

    percentile_file = os.path.join(
        PERCENTILE_DIR,
        f"{wavelet}_swt_percentiles.csv"
    )

    percentiles = pd.read_csv(
        percentile_file,
        parse_dates=["Date"]
    )

    outcomes = pd.read_csv(
        OUTCOME_FILE,
        parse_dates=["Date"]
    )

    df = percentiles.merge(
        outcomes[["Date", "Future_MDD_5D"]],
        on="Date",
        how="inner"
    )

    df = df.dropna(
        subset=["MAS", "Future_MDD_5D"]
    ).copy()

    # Future drawdown stress event
    df["Stress_5D"] = (
        df["Future_MDD_5D"] < STRESS_THRESHOLD
    )

    return df


def analyze_deciles(df):

    # Divide MAS into 10 approximately equal-sized groups
    df["MAS_Decile"] = pd.qcut(
        df["MAS"],
        q=10,
        labels=False,
        duplicates="drop"
    )

    # Unconditional probability of future MDD stress
    baseline_rate = df["Stress_5D"].mean()

    results = []

    for decile, group in df.groupby(
        "MAS_Decile",
        observed=True
    ):

        observations = len(group)

        events = int(
            group["Stress_5D"].sum()
        )

        stress_rate = (
            events / observations
        )

        relative_risk = (
            stress_rate / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        results.append({
            "MAS_Decile": int(decile) + 1,
            "Observations": observations,
            "Stress_Events": events,
            "Stress_Rate": stress_rate,
            "Relative_Risk": relative_risk,
            "MAS_Min": group["MAS"].min(),
            "MAS_Max": group["MAS"].max(),
            "MAS_Median": group["MAS"].median()
        })

    return pd.DataFrame(results), baseline_rate


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    print("=" * 80)
    print("MAS DECILE ANALYSIS")
    print("5-DAY FUTURE DRAWDOWN STRESS: Future_MDD_5D < -3%")
    print("=" * 80)

    for wavelet in WAVELETS:

        print(f"\n{'-' * 80}")
        print(f"WAVELET: {wavelet.upper()}")
        print(f"{'-' * 80}")

        df = load_data(wavelet)

        results, baseline_rate = analyze_deciles(df)

        results.insert(
            0,
            "Wavelet",
            wavelet
        )

        all_results.append(results)

        print(
            f"Observations : {len(df)}"
        )

        print(
            f"Baseline MDD stress rate : "
            f"{baseline_rate * 100:.3f}%"
        )

        print()

        display_table = results.copy()

        display_table["Stress_Rate"] = (
            display_table["Stress_Rate"] * 100
        ).round(3)

        display_table["Relative_Risk"] = (
            display_table["Relative_Risk"]
        ).round(3)

        display_table["MAS_Min"] = (
            display_table["MAS_Min"]
        ).round(2)

        display_table["MAS_Max"] = (
            display_table["MAS_Max"]
        ).round(2)

        display_table["MAS_Median"] = (
            display_table["MAS_Median"]
        ).round(2)

        print(
            display_table[
                [
                    "MAS_Decile",
                    "Observations",
                    "Stress_Events",
                    "Stress_Rate",
                    "Relative_Risk",
                    "MAS_Min",
                    "MAS_Max",
                    "MAS_Median"
                ]
            ].to_string(index=False)
        )

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "mas_decile_5d_mdd_stress.csv"
    )

    final_results.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print(f"Saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()