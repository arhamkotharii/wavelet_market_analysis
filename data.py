import pandas as pd

from config import INPUT_FILE


def load_data():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    required_columns = [
        "Date",
        "Close",
        "Log_Return"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def get_returns(df):

    return df["Log_Return"].to_numpy(
        dtype=float
    )