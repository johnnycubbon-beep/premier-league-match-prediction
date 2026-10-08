"""Inspect low-score dependence ratios for a small historical sample."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "processed" / "full_prem_data.csv"
SEASONS = ["1993-1994", "1994-1995", "1995-1996"]
MAX_GOALS = 9

matches = pd.read_csv(DATA_FILE)
matches = matches[matches["season"].isin(SEASONS)]
home_probabilities = matches["home_goals"].value_counts(normalize=True)
away_probabilities = matches["away_goals"].value_counts(normalize=True)
score_probabilities = matches.groupby(["home_goals", "away_goals"]).size() / len(matches)

ratios = np.full((MAX_GOALS + 1, MAX_GOALS + 1), np.nan)
for home_goals in range(MAX_GOALS + 1):
    for away_goals in range(MAX_GOALS + 1):
        home_marginal = home_probabilities.get(home_goals, 0)
        away_marginal = away_probabilities.get(away_goals, 0)
        if home_marginal and away_marginal:
            joint = score_probabilities.get((home_goals, away_goals), 0)
            ratios[home_goals, away_goals] = joint / (home_marginal * away_marginal)

for score in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(f"{score[0]}-{score[1]}: {ratios[score]:.3f}")
