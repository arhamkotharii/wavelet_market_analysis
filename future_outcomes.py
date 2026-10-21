import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/nifty50_processed.csv"
OUTPUT_DIR = "outcomes"

HORIZONS = [5, 10, 20]


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

    required_columns = [
        "Date",
        "Close",
        "Log_Return"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# FUTURE CUMULATIVE RETURNS
# ============================================================

def calculate_future_returns(returns, horizon):
    """
    Calculate cumulative log return over the NEXT `horizon`
    trading observations.

    R_t,h = r_(t+1) + ... + r_(t+h)

    The current observation r_t is NOT included.
    """

    n = len(returns)

    result = np.full(n, np.nan)

    for t in range(n - horizon):

        future_window = returns[
            t + 1 : t + horizon + 1
        ]

        result[t] = np.sum(future_window)

    return result


# ============================================================
# FUTURE REALIZED VOLATILITY
# ============================================================

def calculate_future_realized_volatility(returns, horizon):
    """
    Calculate realized volatility over the NEXT `horizon`
    trading observations.

    RV_t,h = sqrt(
        (1/h) * sum(r_(t+i)^2)
    )

    The current return r_t is NOT included.
    """

    n = len(returns)

    result = np.full(n, np.nan)

    for t in range(n - horizon):

        future_window = returns[
            t + 1 : t + horizon + 1
        ]

        result[t] = np.sqrt(
            np.mean(future_window ** 2)
        )

    return result


# ============================================================
# FUTURE MAXIMUM DRAWDOWN
# ============================================================

def calculate_future_max_drawdown(returns, horizon):
    """
    Calculate the maximum drawdown over the NEXT `horizon`
    trading observations.

    Returns a negative number.

    Example:

        -0.08 = maximum drawdown of 8%

    We construct a hypothetical wealth path beginning at 1
    immediately after date t.
    """

    n = len(returns)

    result = np.full(n, np.nan)

    for t in range(n - horizon):

        future_returns = returns[
            t + 1 : t + horizon + 1
        ]

        # Convert future log returns into a wealth path.
        wealth = np.exp(
            np.cumsum(future_returns)
        )

        # Include the starting point.
        wealth = np.insert(wealth, 0, 1.0)

        # Running peak.
        running_peak = np.maximum.accumulate(wealth)

        # Drawdown at every point.
        drawdown = (
            wealth / running_peak
        ) - 1.0

        result[t] = np.min(drawdown)

    return result


# ============================================================
# BUILD OUTCOME DATASET
# ============================================================

def build_outcomes(df):

    returns = df["Log_Return"].to_numpy(
        dtype=float
    )

    result = df[
        ["Date", "Close", "Log_Return"]
    ].copy()

    for horizon in HORIZONS:

        print(
            f"Calculating {horizon}-day future return..."
        )

        result[
            f"Future_Return_{horizon}D"
        ] = calculate_future_returns(
            returns,
            horizon
        )

        print(
            f"Calculating {horizon}-day future "
            f"realized volatility..."
        )

        result[
            f"Future_RV_{horizon}D"
        ] = calculate_future_realized_volatility(
            returns,
            horizon
        )

        print(
            f"Calculating {horizon}-day future "
            f"maximum drawdown..."
        )

        result[
            f"Future_MDD_{horizon}D"
        ] = calculate_future_max_drawdown(
            returns,
            horizon
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("FUTURE MARKET OUTCOME CONSTRUCTION")
    print("=" * 70)

    df = load_data()

    print(
        f"Input observations : {len(df)}"
    )

    print(
        f"Start date         : "
        f"{df['Date'].min().date()}"
    )

    print(
        f"End date           : "
        f"{df['Date'].max().date()}"
    )

    result = build_outcomes(df)

    output_path = os.path.join(
        OUTPUT_DIR,
        "nifty50_future_outcomes.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTCOME CONSTRUCTION COMPLETE")
    print("=" * 70)

    print(
        f"Output observations : {len(result)}"
    )

    print(
        f"Saved to            : {output_path}"
    )

    print("\nColumns created:")

    for column in result.columns:
        print(f"  {column}")


if __name__ == "__main__":
    main()