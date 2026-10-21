import os

import numpy as np
import pandas as pd


FEATURE_DIR = "outputs"
PERCENTILE_DIR = "percentiles"

WAVELETS = [
    "haar",
    "db4",
    "sym4",
    "coif1"
]

SWT_FEATURES = [
    "V_D1",
    "V_D2",
    "V_D3",
    "V_A3"
]

PERCENTILE_WINDOW = 1000


def load_swt_features(wavelet):

    filename = (
        f"{wavelet}_swt_rolling_features.csv"
    )

    path = os.path.join(
        FEATURE_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=["Date"]
    )


def historical_percentile(
    current_value,
    history
):
    """
    Percentile rank of the current observation
    relative to historical observations only.

    The current observation is NOT included.
    """

    history = np.asarray(
        history,
        dtype=float
    )

    history = history[
        np.isfinite(history)
    ]

    if len(history) == 0:
        return np.nan

    return (
        np.sum(history < current_value)
        / len(history)
    ) * 100.0


def calculate_percentiles(df):

    result = df[
        ["Date"]
    ].copy()

    for feature in SWT_FEATURES:

        values = df[
            feature
        ].to_numpy(
            dtype=float
        )

        percentiles = np.full(
            len(values),
            np.nan
        )

        for i in range(
            PERCENTILE_WINDOW,
            len(values)
        ):

            history = values[
                i - PERCENTILE_WINDOW:i
            ]

            percentiles[i] = (
                historical_percentile(
                    values[i],
                    history
                )
            )

        result[
            f"Q_{feature.replace('V_', '')}"
        ] = percentiles

    return result


def calculate_mas(result):

    percentile_columns = [
        "Q_D1",
        "Q_D2",
        "Q_D3",
        "Q_A3"
    ]

    result["MAS"] = result[
        percentile_columns
    ].mean(
        axis=1
    )

    return result


def main():

    os.makedirs(
        PERCENTILE_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "SWT CAUSAL PERCENTILE ANALYSIS"
    )
    print("=" * 70)

    for wavelet in WAVELETS:

        print(
            f"\nProcessing SWT + {wavelet}..."
        )

        df = load_swt_features(
            wavelet
        )

        result = calculate_percentiles(
            df
        )

        result = calculate_mas(
            result
        )

        output_path = os.path.join(
            PERCENTILE_DIR,
            f"{wavelet}_swt_percentiles.csv"
        )

        result.to_csv(
            output_path,
            index=False
        )

        valid = result[
            "MAS"
        ].notna()

        print(
            f"Input observations : {len(df)}"
        )

        print(
            f"Valid percentiles  : {valid.sum()}"
        )

        print(
            f"Burn-in observations: "
            f"{(~valid).sum()}"
        )

        print(
            f"Saved to            : "
            f"{output_path}"
        )

    print("\n" + "=" * 70)
    print("PERCENTILE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()