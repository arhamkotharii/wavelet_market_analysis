import os

import pandas as pd

from data import load_data, get_returns
from wavelet import decompose
from features import extract_features
from config import (
    WAVELET,
    TRANSFORM,
    WINDOW,
    LEVELS,
    OUTPUT_DIR
)


def rolling_wavelet_analysis(
    df,
    returns
):
    """
    Apply wavelet decomposition to rolling windows
    and extract wavelet features.

    Returns
    -------
    pandas.DataFrame
        One row per rolling window.
    """

    results = []

    total_windows = len(returns) - WINDOW + 1

    print("\nStarting rolling analysis...")
    print(f"Total windows : {total_windows}")

    for start in range(total_windows):

        end = start + WINDOW

        # ----------------------------------------------
        # Current rolling window
        # ----------------------------------------------

        window = returns[start:end]

        # ----------------------------------------------
        # Wavelet decomposition
        # ----------------------------------------------

        details, approximation = decompose(
            window,
            transform=TRANSFORM,
            wavelet=WAVELET,
            levels=LEVELS
        )

        # ----------------------------------------------
        # Feature extraction
        # ----------------------------------------------

        features = extract_features(
            details,
            approximation
        )

        # ----------------------------------------------
        # Date corresponding to the current observation
        #
        # The window ends at index `end - 1`.
        # ----------------------------------------------

        date = df.iloc[end - 1]["Date"]

        row = {
            "Date": date,
            **features
        }

        results.append(row)

    return pd.DataFrame(results)


def main():

    # ==================================================
    # 1. Load data
    # ==================================================

    df = load_data()
    returns = get_returns(df)

    print("=" * 60)
    print("ROLLING WAVELET ANALYSIS")
    print("=" * 60)

    print(f"Input observations : {len(returns)}")
    print(f"Transform          : {TRANSFORM.upper()}")
    print(f"Wavelet            : {WAVELET}")
    print(f"Rolling window     : {WINDOW}")
    print(f"Decomposition      : {LEVELS} levels")

    # ==================================================
    # 2. Run rolling analysis
    # ==================================================

    results = rolling_wavelet_analysis(
        df,
        returns
    )

    # ==================================================
    # 3. Create output directory
    # ==================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ==================================================
    # 4. Save results
    # ==================================================

    filename = (
        f"{WAVELET}_{TRANSFORM}_"
        f"rolling_features.csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    results.to_csv(
        output_path,
        index=False
    )

    # ==================================================
    # 5. Summary
    # ==================================================

    print("\nAnalysis complete.")

    print(f"Feature observations : {len(results)}")
    print(
        f"Date range           : "
        f"{results['Date'].iloc[0].date()} "
        f"to "
        f"{results['Date'].iloc[-1].date()}"
    )

    print(f"Saved to             : {output_path}")

    print("\nFeature summary:")
    print(
        results[
            [
                "V_D1",
                "V_D2",
                "V_D3",
                "V_A3",
                "HF_LF_ratio",
                "wavelet_entropy"
            ]
        ].describe()
    )

    print("=" * 60)


if __name__ == "__main__":
    main()