import pandas as pd

for wavelet in ["haar", "db4", "sym4", "coif1"]:

    print("\n" + "=" * 80)
    print(f"{wavelet.upper()}")
    print("=" * 80)

    signals = pd.read_csv(
        f"episodes/{wavelet}_signal_episodes.csv",
        parse_dates=["Signal_Start", "Signal_End"]
    )

    stress = pd.read_csv(
        f"episodes/{wavelet}_stress_episodes.csv",
        parse_dates=["Stress_Start", "Stress_End"]
    )

    matches = pd.read_csv(
        f"episodes/{wavelet}_stress_episode_matches.csv",
        parse_dates=[
            "Stress_Start",
            "Stress_End",
            "Signal_Start",
            "Signal_End"
        ]
    )

    print("\nSIGNAL EPISODES")
    print(
        signals[
            [
                "Signal_ID",
                "Signal_Start",
                "Signal_End",
                "Duration_Days",
                "Signal_MAS",
                "Maximum_MAS"
            ]
        ].to_string(index=False)
    )

    print("\nMATCHED STRESS EPISODES")
    print(
        matches[
            [
                "Episode_ID",
                "Stress_Start",
                "Stress_End",
                "Minimum_Future_MDD",
                "Signal_ID",
                "Signal_Start",
                "Signal_End",
                "Signal_MAS",
                "Maximum_MAS",
                "Lead_Time_From_Signal_Start"
            ]
        ].to_string(index=False)
    )