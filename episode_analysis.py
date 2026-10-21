import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PERCENTILE_DIR = "percentiles"
OUTCOME_FILE = "outcomes/nifty50_future_outcomes.csv"
OUTPUT_DIR = "episodes"

WAVELETS = ["haar", "db4", "sym4", "coif1"]

# High-MAS warning threshold
MAS_THRESHOLD = 75.0

# Future stress definition
MDD_THRESHOLD = -0.03

# Maximum number of trading days before stress
# in which a warning signal can occur.
MAX_LOOKBACK = 60


# ============================================================
# DATA LOADING
# ============================================================

def load_data(wavelet):

    percentile_file = os.path.join(
        PERCENTILE_DIR,
        f"{wavelet}_swt_percentiles.csv"
    )

    if not os.path.exists(percentile_file):
        raise FileNotFoundError(
            f"Missing percentile file: {percentile_file}"
        )

    if not os.path.exists(OUTCOME_FILE):
        raise FileNotFoundError(
            f"Missing outcome file: {OUTCOME_FILE}"
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

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Future stress event
    # --------------------------------------------------------

    df["Stress"] = (
        df["Future_MDD_5D"] < MDD_THRESHOLD
    )

    # --------------------------------------------------------
    # High-MAS state
    # --------------------------------------------------------

    df["High_MAS"] = (
        df["MAS"] >= MAS_THRESHOLD
    )

    # --------------------------------------------------------
    # Fresh MAS upward crossing
    #
    # A signal occurs only when MAS moves from:
    #
    #       below 75  -->  75 or above
    #
    # This prevents a long high-MAS regime from being counted
    # as many separate warning signals.
    # --------------------------------------------------------

    df["Signal_Crossing"] = (
        df["High_MAS"] &
        ~df["High_MAS"].shift(
            1,
            fill_value=False
        )
    )

    return df


# ============================================================
# CONTIGUOUS RUN DETECTION
# ============================================================

def find_runs(boolean_series):

    """
    Identify contiguous True runs.

    Returns:
        list of tuples:
        (start_index, end_index)
    """

    values = boolean_series.to_numpy()

    runs = []

    start = None

    for i, value in enumerate(values):

        if value and start is None:
            start = i

        elif not value and start is not None:

            runs.append(
                (start, i - 1)
            )

            start = None

    # Handle a run continuing to the end
    if start is not None:

        runs.append(
            (start, len(values) - 1)
        )

    return runs


# ============================================================
# STRESS EPISODES
# ============================================================

def build_stress_episodes(df):

    stress_runs = find_runs(
        df["Stress"]
    )

    episodes = []

    for episode_id, (start, end) in enumerate(
        stress_runs,
        start=1
    ):

        episode = df.iloc[
            start:end + 1
        ]

        episodes.append({

            "Episode_ID":
                episode_id,

            "Stress_Start_Index":
                start,

            "Stress_End_Index":
                end,

            "Stress_Start":
                df.loc[start, "Date"],

            "Stress_End":
                df.loc[end, "Date"],

            "Duration_Days":
                end - start + 1,

            "Minimum_Future_MDD":
                episode[
                    "Future_MDD_5D"
                ].min()
        })

    return pd.DataFrame(
        episodes
    )


# ============================================================
# SIGNAL EPISODES
# ============================================================

def build_signal_episodes(df):

    signal_indices = df.index[
        df["Signal_Crossing"]
    ].tolist()

    signals = []

    for signal_id, signal_index in enumerate(
        signal_indices,
        start=1
    ):

        # Find the end of this high-MAS regime.
        #
        # Since Signal_Crossing identifies the beginning
        # of a high-MAS regime, continue until MAS falls
        # below the threshold.

        end_index = signal_index

        while (
            end_index + 1 < len(df)
            and df.loc[
                end_index + 1,
                "High_MAS"
            ]
        ):
            end_index += 1

        regime = df.loc[
            signal_index:end_index
        ]

        signals.append({

            "Signal_ID":
                signal_id,

            "Signal_Start_Index":
                signal_index,

            "Signal_End_Index":
                end_index,

            "Signal_Start":
                df.loc[
                    signal_index,
                    "Date"
                ],

            "Signal_End":
                df.loc[
                    end_index,
                    "Date"
                ],

            "Duration_Days":
                end_index - signal_index + 1,

            "Signal_MAS":
                df.loc[
                    signal_index,
                    "MAS"
                ],

            "Maximum_MAS":
                regime["MAS"].max(),

            "Mean_MAS":
                regime["MAS"].mean()
        })

    return pd.DataFrame(
        signals
    )


# ============================================================
# MATCH SIGNALS TO FUTURE STRESS
# ============================================================

def match_signals_to_stress(
    df,
    stress_episodes,
    signal_episodes
):

    matches = []

    for _, stress in stress_episodes.iterrows():

        stress_start = int(
            stress[
                "Stress_Start_Index"
            ]
        )

        # ----------------------------------------------------
        # Search window:
        #
        # [stress_start - 60, stress_start - 1]
        #
        # Signal must occur BEFORE stress begins.
        # ----------------------------------------------------

        lookback_start = max(
            0,
            stress_start - MAX_LOOKBACK
        )

        candidates = signal_episodes[
            (
                signal_episodes[
                    "Signal_Start_Index"
                ] >= lookback_start
            )
            &
            (
                signal_episodes[
                    "Signal_Start_Index"
                ] < stress_start
            )
        ].copy()

        if len(candidates) == 0:
            continue

        # ----------------------------------------------------
        # Use the MOST RECENT signal crossing before stress.
        #
        # This is the most natural interpretation of:
        #
        # "Was there a fresh warning signal before stress?"
        # ----------------------------------------------------

        signal = candidates.sort_values(
            "Signal_Start_Index"
        ).iloc[-1]

        signal_start = int(
            signal[
                "Signal_Start_Index"
            ]
        )

        signal_end = int(
            signal[
                "Signal_End_Index"
            ]
        )

        # Lead time from the moment the warning began
        lead_time_start = (
            stress_start -
            signal_start
        )

        # Lead time from the end of the high-MAS regime
        lead_time_end = (
            stress_start -
            signal_end
        )

        # Safety check
        if lead_time_start > MAX_LOOKBACK:
            continue

        matches.append({

            "Episode_ID":
                stress["Episode_ID"],

            "Stress_Start":
                stress["Stress_Start"],

            "Stress_End":
                stress["Stress_End"],

            "Stress_Duration":
                stress["Duration_Days"],

            "Minimum_Future_MDD":
                stress["Minimum_Future_MDD"],

            "Signal_ID":
                signal["Signal_ID"],

            "Signal_Start":
                signal["Signal_Start"],

            "Signal_End":
                signal["Signal_End"],

            "Signal_Duration":
                signal["Duration_Days"],

            "Signal_MAS":
                signal["Signal_MAS"],

            "Maximum_MAS":
                signal["Maximum_MAS"],

            "Mean_MAS":
                signal["Mean_MAS"],

            "Lead_Time_From_Signal_Start":
                lead_time_start,

            "Lead_Time_From_Signal_End":
                lead_time_end
        })

    return pd.DataFrame(
        matches
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary(
    df,
    stress_episodes,
    signal_episodes,
    matches
):

    total_stress = len(
        stress_episodes
    )

    total_signals = len(
        signal_episodes
    )

    matched = len(
        matches
    )

    if total_stress > 0:

        coverage = (
            matched /
            total_stress
        )

    else:

        coverage = np.nan

    # --------------------------------------------------------
    # Signal precision
    #
    # Number of signal episodes that were followed by
    # a stress episode within MAX_LOOKBACK days.
    # --------------------------------------------------------

    if matched > 0:

        useful_signals = (
            matches["Signal_ID"]
            .nunique()
        )

    else:

        useful_signals = 0

    if total_signals > 0:

        precision = (
            useful_signals /
            total_signals
        )

    else:

        precision = np.nan

    # --------------------------------------------------------
    # Lead-time statistics
    # --------------------------------------------------------

    if matched > 0:

        median_start = (
            matches[
                "Lead_Time_From_Signal_Start"
            ].median()
        )

        mean_start = (
            matches[
                "Lead_Time_From_Signal_Start"
            ].mean()
        )

        median_end = (
            matches[
                "Lead_Time_From_Signal_End"
            ].median()
        )

        mean_end = (
            matches[
                "Lead_Time_From_Signal_End"
            ].mean()
        )

        min_start = (
            matches[
                "Lead_Time_From_Signal_Start"
            ].min()
        )

        max_start = (
            matches[
                "Lead_Time_From_Signal_Start"
            ].max()
        )

    else:

        median_start = np.nan
        mean_start = np.nan
        median_end = np.nan
        mean_end = np.nan
        min_start = np.nan
        max_start = np.nan

    return {

        "Wavelet": None,

        "Observations":
            len(df),

        "Stress_Episodes":
            total_stress,

        "Signal_Episodes":
            total_signals,

        "Matched_Stress_Episodes":
            matched,

        "Stress_Episode_Coverage":
            coverage,

        "Useful_Signal_Episodes":
            useful_signals,

        "Signal_Precision":
            precision,

        "Median_Lead_From_Signal_Start":
            median_start,

        "Mean_Lead_From_Signal_Start":
            mean_start,

        "Minimum_Lead_From_Signal_Start":
            min_start,

        "Maximum_Lead_From_Signal_Start":
            max_start,

        "Median_Lead_From_Signal_End":
            median_end,

        "Mean_Lead_From_Signal_End":
            mean_end
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    summaries = []

    print("=" * 80)
    print("INDEPENDENT STRESS EPISODE ANALYSIS")
    print("=" * 80)

    print(
        f"Stress definition: "
        f"Future_MDD_5D < {MDD_THRESHOLD:.0%}"
    )

    print(
        f"Signal definition: "
        f"MAS upward crossing of {MAS_THRESHOLD:.0f}"
    )

    print(
        f"Maximum signal lookback: "
        f"{MAX_LOOKBACK} trading days"
    )

    for wavelet in WAVELETS:

        print("\n" + "-" * 80)
        print(
            f"WAVELET: {wavelet.upper()}"
        )
        print("-" * 80)

        df = load_data(
            wavelet
        )

        stress_episodes = (
            build_stress_episodes(df)
        )

        signal_episodes = (
            build_signal_episodes(df)
        )

        matches = (
            match_signals_to_stress(
                df,
                stress_episodes,
                signal_episodes
            )
        )

        summary = calculate_summary(
            df,
            stress_episodes,
            signal_episodes,
            matches
        )

        summary["Wavelet"] = wavelet

        summaries.append(
            summary
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"Observations: "
            f"{len(df)}"
        )

        print(
            f"Independent stress episodes: "
            f"{len(stress_episodes)}"
        )

        print(
            f"Fresh MAS signal episodes: "
            f"{len(signal_episodes)}"
        )

        print(
            f"Stress episodes preceded by signal: "
            f"{len(matches)}"
        )

        print(
            f"Stress episode coverage: "
            f"{summary['Stress_Episode_Coverage'] * 100:.2f}%"
        )

        print(
            f"Signal precision: "
            f"{summary['Signal_Precision'] * 100:.2f}%"
        )

        if len(matches) > 0:

            print(
                f"Median lead time "
                f"(signal start): "
                f"{summary['Median_Lead_From_Signal_Start']:.1f} "
                f"trading days"
            )

            print(
                f"Mean lead time "
                f"(signal start): "
                f"{summary['Mean_Lead_From_Signal_Start']:.1f} "
                f"trading days"
            )

            print(
                f"Lead-time range: "
                f"{summary['Minimum_Lead_From_Signal_Start']:.0f} "
                f"to "
                f"{summary['Maximum_Lead_From_Signal_Start']:.0f} "
                f"days"
            )

            print(
                f"Median lead time "
                f"(signal end): "
                f"{summary['Median_Lead_From_Signal_End']:.1f} "
                f"trading days"
            )

        # ----------------------------------------------------
        # Save stress episodes
        # ----------------------------------------------------

        stress_file = os.path.join(
            OUTPUT_DIR,
            f"{wavelet}_stress_episodes.csv"
        )

        stress_episodes.to_csv(
            stress_file,
            index=False
        )

        # ----------------------------------------------------
        # Save signal episodes
        # ----------------------------------------------------

        signal_file = os.path.join(
            OUTPUT_DIR,
            f"{wavelet}_signal_episodes.csv"
        )

        signal_episodes.to_csv(
            signal_file,
            index=False
        )

        # ----------------------------------------------------
        # Save matched episodes
        # ----------------------------------------------------

        match_file = os.path.join(
            OUTPUT_DIR,
            f"{wavelet}_stress_episode_matches.csv"
        )

        matches.to_csv(
            match_file,
            index=False
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "episode_analysis_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("EPISODE ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()