"""Tune Elo parameters and initialization policies with a Poisson GLM.

The grid is selected on 2018-19 and 2019-20 validation matches. Matches from
2020-21 onward are deliberately excluded so they remain available for a final
out-of-sample evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = ROOT / "Data" / "processed"
OUTPUT_DIR = ROOT / "Output"
MATCH_FILE = PROCESSED_DATA / "full_prem_data.csv"
FORM_FILE = PROCESSED_DATA / "prem_form_guide.csv"
RELEGATED_FILE = PROCESSED_DATA / "relegated.json"
RESULTS_FILE = OUTPUT_DIR / "elo_glm_tuning.csv"

K_VALUES = [5, 10, 15, 20, 25, 30, 40]
DISCOUNTS = [0, 50, 100, 150, 200, 250, 300]
PROMOTED_START_METHODS = ["fixed_1500", "mean_ratings_minus_d"]
RELEGATED_POLICIES = ["remove", "retain"]

TRAIN_END_SEASON = "2018-2019"
VALIDATION_END_SEASON = "2020-2021"
GOAL_FEATURES = ["ELO_diff", "Form_diff", "Attack_diff", "Defense_diff"]
GOAL_FORMULA = "{goals} ~ ELO_diff + Form_diff + Attack_diff + Defense_diff"


def calculate_elo(
    matches: pd.DataFrame,
    relegated_teams: dict[str, list[str]],
    k_factor: int,
    discount: int,
    promoted_start_method: str,
    relegated_policy: str,
) -> pd.DataFrame:
    """Return each match's pre-match Elo values for one parameter set."""
    season_starts = set(matches.groupby("season", sort=False)["Match ID"].first().iloc[1:])
    ratings: dict[str, float] = {}
    home_ratings: list[float] = []
    away_ratings: list[float] = []
    first_season = matches["season"].iloc[0]

    for match_index, row in enumerate(matches.itertuples(index=False)):
        if match_index in season_starts and relegated_policy == "remove":
            previous_season = matches.iloc[match_index - 1]["season"]
            for team in relegated_teams.get(previous_season, []):
                ratings.pop(team, None)

        def get_rating(team: str) -> float:
            if team in ratings:
                return ratings[team]
            if row.season == first_season or promoted_start_method == "fixed_1500":
                return 1500.0
            league_average = sum(ratings.values()) / len(ratings) if ratings else 1500.0
            return league_average - discount

        home_elo = get_rating(row.home_team)
        away_elo = get_rating(row.away_team)
        home_ratings.append(home_elo)
        away_ratings.append(away_elo)

        home_expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        home_actual = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        ratings[row.home_team] = home_elo + k_factor * (home_actual - home_expected)
        ratings[row.away_team] = away_elo + k_factor * (
            (1 - home_actual) - (1 - home_expected)
        )

    return matches[["Match ID"]].assign(
        **{"Home ELO": home_ratings, "Away ELO": away_ratings}
    )


def outcome_probabilities(home_rate: float, away_rate: float) -> dict[str, float]:
    """Convert independent Poisson goal rates into match outcome probabilities."""
    largest_rate = max(home_rate, away_rate)
    goal_limit = max(10, int(np.ceil(largest_rate + 10 * np.sqrt(largest_rate))))
    goals = np.arange(goal_limit)
    home_probs = poisson.pmf(goals, home_rate)
    away_probs = poisson.pmf(goals, away_rate)
    away_cumulative = np.cumsum(away_probs)

    return {
        "H": float(np.sum(home_probs * np.concatenate(([0.0], away_cumulative[:-1])))),
        "D": float(np.sum(home_probs * away_probs)),
        "A": float(np.sum(home_probs * (away_cumulative[-1] - away_cumulative))),
    }


def evaluate_configuration(
    matches: pd.DataFrame,
    form_guide: pd.DataFrame,
    relegated_teams: dict[str, list[str]],
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    k_factor: int,
    discount: int | None,
    promoted_start_method: str,
    relegated_policy: str,
) -> dict[str, object]:
    elo = calculate_elo(
        matches=matches,
        relegated_teams=relegated_teams,
        k_factor=k_factor,
        discount=discount or 0,
        promoted_start_method=promoted_start_method,
        relegated_policy=relegated_policy,
    )
    data = form_guide.drop(columns=["Home ELO", "Away ELO"], errors="ignore").merge(
        elo, on="Match ID", validate="one_to_one"
    )
    data["ELO_diff"] = data["Home ELO"] - data["Away ELO"]
    data["Form_diff"] = data["Home Form"] - data["Away Form"]
    data["Attack_diff"] = data["Home Attack Form"] - data["Away Attack Form"]
    data["Defense_diff"] = data["Home Defense Form"] - data["Away Defense Form"]

    # Match IDs and form rows share chronological order. Restrict model fit and
    # scoring to the same fixed temporal windows for every Elo configuration.
    train = data.iloc[: len(train_data)]
    validation = data.iloc[len(train_data) : len(train_data) + len(validation_data)]
    home_model = smf.glm(
        formula=GOAL_FORMULA.format(goals="home_goals"),
        data=train,
        family=sm.families.Poisson(),
    ).fit()
    away_model = smf.glm(
        formula=GOAL_FORMULA.format(goals="away_goals"),
        data=train,
        family=sm.families.Poisson(),
    ).fit()

    home_rates = home_model.predict(validation[GOAL_FEATURES])
    away_rates = away_model.predict(validation[GOAL_FEATURES])
    predictions = []
    for home_rate, away_rate in zip(home_rates, away_rates):
        probabilities = outcome_probabilities(home_rate, away_rate)
        predictions.append(max(probabilities, key=probabilities.get))

    return {
        "K": k_factor,
        "promoted_start_method": promoted_start_method,
        "d": discount if promoted_start_method == "mean_ratings_minus_d" else None,
        "relegated_policy": relegated_policy,
        "validation_accuracy": accuracy_score(validation["Result"], predictions),
        "validation_matches": len(validation),
    }


def main() -> None:
    matches = pd.read_csv(MATCH_FILE).sort_values("Match ID").reset_index(drop=True)
    form_guide = pd.read_csv(FORM_FILE).sort_values("Match ID").reset_index(drop=True)
    with RELEGATED_FILE.open(encoding="utf-8") as file:
        relegated_teams = json.load(file)

    train_end = form_guide.index[form_guide["Season"].eq(TRAIN_END_SEASON)][0]
    validation_end = form_guide.index[form_guide["Season"].eq(VALIDATION_END_SEASON)][0]
    if validation_end <= train_end:
        raise ValueError("Validation end must follow the training end season.")
    train_data = form_guide.iloc[:train_end]
    validation_data = form_guide.iloc[train_end:validation_end]
    if train_data.empty or validation_data.empty:
        raise ValueError("Training and validation windows must both contain matches.")

    results = []
    for k_factor in K_VALUES:
        for start_method in PROMOTED_START_METHODS:
            discounts = DISCOUNTS if start_method == "mean_ratings_minus_d" else [None]
            for discount in discounts:
                for relegated_policy in RELEGATED_POLICIES:
                    results.append(
                        evaluate_configuration(
                            matches=matches,
                            form_guide=form_guide,
                            relegated_teams=relegated_teams,
                            train_data=train_data,
                            validation_data=validation_data,
                            k_factor=k_factor,
                            discount=discount,
                            promoted_start_method=start_method,
                            relegated_policy=relegated_policy,
                        )
                    )

    results_frame = pd.DataFrame(results).sort_values(
        ["validation_accuracy", "K"], ascending=[False, True]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(RESULTS_FILE, index=False)
    print(
        f"Validation seasons: 2018-19 and 2019-20 ({len(validation_data):,} matches); "
        "2020-21 onward was not scored."
    )
    print(f"Evaluated {len(results_frame)} Elo configurations. Results: {RESULTS_FILE}")
    print("\nTop 10 configurations:")
    print(results_frame.head(10).to_string(index=False, formatters={"validation_accuracy": "{:.2%}".format}))


if __name__ == "__main__":
    main()
