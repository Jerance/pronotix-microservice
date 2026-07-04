import pandas as pd


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    ]
    df_prepared = df[required_columns].copy()

    df_prepared.loc[:, "is_home_winner"] = (
        df_prepared["home_score"] > df_prepared["away_score"]
    ).astype(int)
    df_prepared.loc[:, "is_away_winner"] = (
        df_prepared["away_score"] > df_prepared["home_score"]
    ).astype(int)

    return df_prepared
