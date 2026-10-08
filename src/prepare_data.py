"""Build the processed Premier League match and pre-match feature datasets."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "Data" / "raw"
PROCESSED_DATA = ROOT / "Data" / "processed"
RESULTS_FILE = RAW_DATA / "results.csv"
FORM_FILE = PROCESSED_DATA / "prem_form_guide.csv"
MATCH_FILE = PROCESSED_DATA / "full_prem_data.csv"
RELEGATED_FILE = PROCESSED_DATA / "relegated.json"

FORM_FEATURES = {
    "Points": ("Form", 6),
    "Goals For": ("Attack Form", 10),
    "Goals Against": ("Defense Form", 10),
}


def _calculate_team_form(matches: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Calculate venue-specific rolling features using past matches only."""
    matches = matches.sort_values(["Team", "Match ID"]).copy()
    grouped = matches.groupby("Team", sort=False)
    for source, (feature, window) in FORM_FEATURES.items():
        matches[f"{prefix} {feature}"] = grouped[source].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
    return matches


def _calculate_elo(matches: pd.DataFrame, relegated: dict) -> pd.DataFrame:
    """Calculate pre-match Elo ratings, discounting newly promoted teams."""
    season_starts = set(
        matches.groupby("season", sort=False)["Match ID"].first().iloc[1:]
    )
    ratings = {}
    home_ratings, away_ratings = [], []
    k_factor, promotion_discount = 20, 150
    first_season = matches["season"].iloc[0]

    for match_index, row in enumerate(matches.itertuples(index=False)):
        if match_index in season_starts:
            previous_season = matches.iloc[match_index - 1]["season"]
            for team in relegated.get(previous_season, []):
                ratings.pop(team, None)

        def rating(team):
            if team in ratings:
                return ratings[team]
            if row.season == first_season:
                return 1500.0
            return (sum(ratings.values()) / len(ratings) - promotion_discount) if ratings else 1350.0

        home_elo = rating(row.home_team)
        away_elo = rating(row.away_team)
        home_ratings.append(home_elo)
        away_ratings.append(away_elo)

        home_expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        home_actual = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        ratings[row.home_team] = home_elo + k_factor * (home_actual - home_expected)
        ratings[row.away_team] = away_elo + k_factor * ((1 - home_actual) - (1 - home_expected))

    return matches.assign(**{"Home ELO": home_ratings, "Away ELO": away_ratings})


def main() -> None:
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    season_files = sorted(RAW_DATA.glob("????-????.csv"))
    if season_files:
        season_frames = []
        for season_file in season_files:
            season = season_file.stem
            season_frame = pd.read_csv(
                season_file,
                usecols=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"],
                encoding="latin1",
            )
            season_frame["season"] = season
            season_frames.append(season_frame)
        matches = pd.concat(season_frames, ignore_index=True).rename(
            columns={
                "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "home_goals",
                "FTAG": "away_goals", "FTR": "result",
            }
        )
        matches = matches.dropna(subset=["home_team", "away_team", "home_goals", "away_goals", "result"])
    else:
        matches = pd.read_csv(RESULTS_FILE)
    matches["Match ID"] = range(len(matches))

    with RELEGATED_FILE.open(encoding="utf-8") as file:
        relegated = json.load(file)

    matches = _calculate_elo(matches, relegated)
    matches.to_csv(MATCH_FILE, index=False)

    home = matches[
        ["Match ID", "season", "home_team", "home_goals", "away_goals", "result"]
    ].rename(
        columns={
            "season": "Season", "home_team": "Team", "home_goals": "Goals For",
            "away_goals": "Goals Against", "result": "Result",
        }
    )
    home["Outcome"] = home["Result"].map({"H": "W", "D": "D", "A": "L"})
    home["Venue"] = "Home"

    away = matches[
        ["Match ID", "season", "away_team", "home_goals", "away_goals", "result"]
    ].rename(
        columns={
            "season": "Season", "away_team": "Team", "home_goals": "Goals Against",
            "away_goals": "Goals For", "result": "Result",
        }
    )
    away["Outcome"] = away["Result"].map({"H": "L", "D": "D", "A": "W"})
    away["Venue"] = "Away"

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches["Points"] = team_matches["Outcome"].map({"W": 3, "D": 1, "L": 0})
    home_matches = _calculate_team_form(team_matches[team_matches["Venue"] == "Home"], "Home")
    away_matches = _calculate_team_form(team_matches[team_matches["Venue"] == "Away"], "Away")

    home_columns = ["Match ID", "Season", "Team", "Home Form", "Home Attack Form", "Home Defense Form", "Result"]
    away_columns = ["Match ID", "Season", "Team", "Away Form", "Away Attack Form", "Away Defense Form", "Result"]
    form_guide = home_matches[home_columns].rename(columns={"Team": "Home Team"}).merge(
        away_matches[away_columns].rename(columns={"Team": "Away Team"}),
        on=["Match ID", "Season", "Result"],
        validate="one_to_one",
    )
    form_guide = form_guide.merge(
        matches[["Match ID", "Home ELO", "Away ELO", "home_goals", "away_goals"]],
        on="Match ID",
        validate="one_to_one",
    ).sort_values("Match ID")
    form_guide = form_guide.dropna(
        subset=["Home Form", "Home Attack Form", "Home Defense Form", "Away Form", "Away Attack Form", "Away Defense Form"]
    ).reset_index(drop=True)

    form_guide["Basic Pred"] = form_guide["Home Form"].gt(form_guide["Away Form"]).map(
        {True: "H", False: "A"}
    )
    form_guide["Correct"] = form_guide["Basic Pred"].eq(form_guide["Result"]).astype(int)
    form_guide.to_csv(FORM_FILE, index=False)
    print(f"Saved {len(matches):,} matches to {MATCH_FILE}")
    print(f"Saved {len(form_guide):,} feature rows to {FORM_FILE}")


if __name__ == "__main__":
    main()
