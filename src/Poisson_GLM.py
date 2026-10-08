"""Fit independent home and away Poisson goal models and evaluate outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "processed" / "prem_form_guide.csv"
FORMULA = "{goals} ~ ELO_diff + Form_diff + Attack_diff + Defense_diff"
FEATURES = ["ELO_diff", "Form_diff", "Attack_diff", "Defense_diff"]


def outcome_probabilities(home_rate: float, away_rate: float) -> dict[str, float]:
    """Return truncated Poisson probabilities for home win, draw, and away win."""
    goal_limit = max(10, int(np.ceil(max(home_rate, away_rate) + 10 * np.sqrt(max(home_rate, away_rate)))))
    goals = np.arange(goal_limit)
    home_probs = poisson.pmf(goals, home_rate)
    away_probs = poisson.pmf(goals, away_rate)
    away_cumulative = np.cumsum(away_probs)

    return {
        "H": float(np.sum(home_probs * np.concatenate(([0.0], away_cumulative[:-1])))),
        "D": float(np.sum(home_probs * away_probs)),
        "A": float(np.sum(home_probs * (away_cumulative[-1] - away_cumulative))),
    }


form_guide = pd.read_csv(DATA_FILE)
form_guide["ELO_diff"] = form_guide["Home ELO"] - form_guide["Away ELO"]
form_guide["Form_diff"] = form_guide["Home Form"] - form_guide["Away Form"]
form_guide["Attack_diff"] = form_guide["Home Attack Form"] - form_guide["Away Attack Form"]
form_guide["Defense_diff"] = form_guide["Home Defense Form"] - form_guide["Away Defense Form"]

train_season = "2016-2017"
validation_season = "2018-2019"
train_end = form_guide[form_guide["Season"] == train_season].index[0]
validation_end = form_guide[form_guide["Season"] == validation_season].index[0]
train_data = form_guide.iloc[:train_end]
validation_data = form_guide.iloc[train_end:validation_end]

home_model = smf.glm(
    formula=FORMULA.format(goals="home_goals"),
    data=train_data,
    family=sm.families.Poisson(),
).fit()
away_model = smf.glm(
    formula=FORMULA.format(goals="away_goals"),
    data=train_data,
    family=sm.families.Poisson(),
).fit()

home_rates = home_model.predict(validation_data[FEATURES])
away_rates = away_model.predict(validation_data[FEATURES])
predictions = []
for home_rate, away_rate in zip(home_rates, away_rates):
    probabilities = outcome_probabilities(home_rate, away_rate)
    predictions.append(max(probabilities, key=probabilities.get))
accuracy = accuracy_score(validation_data["Result"], predictions)
print(f"Validation accuracy ({train_season} to {validation_season}): {accuracy:.2%}")
