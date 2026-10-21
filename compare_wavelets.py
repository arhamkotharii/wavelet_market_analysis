import os

import numpy as np
import pandas as pd


# ============================================================
# DIRECTORIES
# ============================================================

FEATURE_DIR = "outputs"
COMPARISON_DIR = "comparison"


# ============================================================
# EXPERIMENTAL GRID
# ============================================================

WAVELETS = [
    "haar",
    "db4",
    "sym4",
    "coif1"
]

TRANSFORMS = [
    "dwt",
    "swt"
]


# ============================================================
# FEATURES TO COMPARE
# ============================================================

FEATURES = [
    "V_D1",
    "V_D2",
    "V_D3",
    "V_A3",
    "HF_LF_ratio",
    "wavelet_entropy"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_wavelet_data(
    transform,
    wavelet
):

    filename = (
        f"{wavelet}_{transform}_"
        f"rolling_features.csv"
    )

    path = os.path.join(
        FEATURE_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"]
    )

    return df


# ============================================================
# RELATIVE DAY-TO-DAY CHANGE
# ============================================================

def relative_day_change(series):

    previous = series.shift(1)

    denominator = previous.abs()

    change = (
        (series - previous).abs()
        / denominator
    )

    change = change.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return change.dropna()


# ============================================================
# STABILITY STATISTICS
# ============================================================

def calculate_stability(
    df,
    feature
):

    changes = relative_day_change(
        df[feature]
    )

    return {
        "median": changes.quantile(0.50),
        "p75": changes.quantile(0.75),
        "p90": changes.quantile(0.90),
        "p95": changes.quantile(0.95),
        "p99": changes.quantile(0.99),
        "max": changes.max()
    }


# ============================================================
# DATE ALIGNMENT CHECK
# ============================================================

def check_date_alignment(
    datasets
):

    reference_name = list(
        datasets.keys()
    )[0]

    reference_dates = datasets[
        reference_name
    ]["Date"]

    for name, df in datasets.items():

        if not reference_dates.equals(
            df["Date"]
        ):

            raise ValueError(
                f"Date alignment failed "
                f"for {name}."
            )

    print(
        "\nDate alignment check : PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "WAVELET TRANSFORM + BASIS "
        "STABILITY COMPARISON"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Create comparison directory
    # --------------------------------------------------------

    os.makedirs(
        COMPARISON_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load all 8 datasets first
    # --------------------------------------------------------

    datasets = {}

    for transform in TRANSFORMS:

        for wavelet in WAVELETS:

            name = (
                f"{wavelet}_{transform}"
            )

            print(
                f"\nLoading "
                f"{transform.upper()} + {wavelet}..."
            )

            df = load_wavelet_data(
                transform,
                wavelet
            )

            print(
                f"Observations : {len(df)}"
            )

            # --------------------------------------------
            # Observation count check
            # --------------------------------------------

            if len(df) != 3878:

                raise ValueError(
                    f"{name} contains "
                    f"{len(df)} observations. "
                    f"Expected 3878."
                )

            # --------------------------------------------
            # Required columns check
            # --------------------------------------------

            required_columns = [
                "Date",
                *FEATURES
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"{name} is missing columns: "
                    f"{missing_columns}"
                )

            datasets[name] = df

    # --------------------------------------------------------
    # Check that every configuration has identical dates
    # --------------------------------------------------------

    check_date_alignment(
        datasets
    )

    # --------------------------------------------------------
    # Calculate stability
    # --------------------------------------------------------

    all_results = []

    for transform in TRANSFORMS:

        for wavelet in WAVELETS:

            name = (
                f"{wavelet}_{transform}"
            )

            df = datasets[name]

            for feature in FEATURES:

                stats = calculate_stability(
                    df,
                    feature
                )

                row = {
                    "Transform": transform,
                    "Wavelet": wavelet,
                    "Feature": feature,
                    **stats
                }

                all_results.append(row)

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    detailed_path = os.path.join(
        COMPARISON_DIR,
        "wavelet_transform_stability_comparison.csv"
    )

    results.to_csv(
        detailed_path,
        index=False
    )

    # ========================================================
    # PRINT RESULTS FEATURE BY FEATURE
    # ========================================================

    for feature in FEATURES:

        print("\n")
        print("=" * 80)
        print(feature)
        print("=" * 80)

        subset = results[
            results["Feature"] == feature
        ].copy()

        table = subset.pivot(
            index="Wavelet",
            columns="Transform",
            values=[
                "median",
                "p90",
                "p95",
                "p99"
            ]
        )

        print(
            table.to_string(
                float_format=lambda x:
                f"{x:.4f}"
            )
        )

    # ========================================================
    # AVERAGE SCALE-VARIANCE INSTABILITY
    # ========================================================

    scale_features = [
        "V_D1",
        "V_D2",
        "V_D3",
        "V_A3"
    ]

    scale_results = results[
        results["Feature"].isin(
            scale_features
        )
    ]

    average_scale_stability = (
        scale_results
        .groupby(
            [
                "Transform",
                "Wavelet"
            ]
        )[
            [
                "median",
                "p75",
                "p90",
                "p95",
                "p99"
            ]
        ]
        .mean()
    )

    print("\n")
    print("=" * 80)
    print(
        "AVERAGE SCALE-VARIANCE "
        "INSTABILITY"
    )
    print("=" * 80)

    print(
        average_scale_stability.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE AVERAGE RESULTS
    # ========================================================

    average_path = os.path.join(
        COMPARISON_DIR,
        "wavelet_transform_average_stability.csv"
    )

    average_scale_stability.to_csv(
        average_path
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"Detailed results : {detailed_path}"
    )

    print(
        f"Average results  : {average_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()