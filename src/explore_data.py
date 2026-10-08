"""Print basic summaries of the processed Premier League match data."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "processed" / "full_prem_data.csv"
DATA_FILE_2 = ROOT / "Data" / "processed" / "prem_form_guide.csv"


matches = pd.read_csv(DATA_FILE)

# print(f"Matches: {len(matches):,}")
# print(f"Seasons: {matches['season'].min()} to {matches['season'].max()}")
# print("Results by season (%):")
# print(matches.groupby("season")["result"].value_counts(normalize=True).unstack(fill_value=0).mul(100).round(1))

# goal_summary = matches[["home_goals", "away_goals"]].agg(["mean", "median", "max"])
# print("\nGoal summary:")
# print(goal_summary)

form = pd.read_csv(DATA_FILE_2)
print(form.sort_values(["Home ELO"])[["Season","Home Team", "Home ELO"]])
print(form["Home Team"].unique())
print(form.groupby("Season")[["Home ELO","Away ELO"]].mean())
print(form[form["Home Team"]=="Tottenham"][["Home Team","Home ELO","Season"]].sort_values("Home ELO"))


