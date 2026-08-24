import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import numpy as np

# Read in the CSV file to a pandas dataframe type
df = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/raw/results.csv")

# Setting column with match ID helps to get around indices getting messed up
df["Match ID"] = range(len(df))


# Gather all the home games for each team so change result column to Win, Lose or Draw
# Add a Venue column to keep track once we merge Home and Away games for each team
home = df[["Match ID","season", "home_team", "home_goals", "away_goals", "result"]].copy()
home = home.rename(columns={"season":"Season",
                            "home_team": "Team",
                            "home_goals": "Goals For",
                            "away_goals": "Goals Against",
                            "result": "Result"
                            })
home["Outcome"] = home["Result"].map({
    "H": "W",
    "D": "D",
    "A": "L"
})
home["Venue"] = "Home"

# Gather all the away games for each team so change result column to Win, Lose or Draw
# Add a Venue column to keep track once we merge Home and Away games for each team
away = df[["Match ID","season","away_team","home_goals", "away_goals", "result"]].copy()
away = away.rename(columns={"season": "Season",
                            "away_team": "Team",
                            "home_goals": "Goals Against",
                            "away_goals": "Goals For",
                            "result": "Result"
                            })
away["Outcome"] = away["Result"].map({
    'H':'L',
    'D':'D',
    'A':'W'
    })
away["Venue"] = "Away"

# Concatenate home and away frames so that we now have double the rows and two rows corresponding to each match
# Map Win Lose Draw to the points values of each
all_matches = pd.concat([home, away], ignore_index=True)
all_matches["Points"] = all_matches["Outcome"].map({'W': 3, 'D': 1, 'L': 0})
all_matches = all_matches.sort_values(["Team","Match ID"])

# Re split the home and away matches (last code was somewhat unnecessary)
home_matches = all_matches[all_matches["Venue"]=="Home"].sort_values(["Team","Match ID"])
away_matches = all_matches[all_matches["Venue"]=="Away"].sort_values(["Team","Match ID"])

previous_home_points = home_matches.groupby("Team")[["Points","Goals For","Goals Against"]].shift(1)
home_form = previous_home_points["Points"].rolling(6).mean()
home_attack_form = previous_home_points["Goals For"].rolling(10).mean()
home_defense_form = previous_home_points["Goals Against"].rolling(10).mean()




previous_away_points = away_matches.groupby("Team")[["Points","Goals For","Goals Against"]].shift(1)
away_form = previous_away_points["Points"].rolling(6).mean()
away_attack_form = previous_away_points["Goals For"].rolling(10).mean()
away_defense_form = previous_away_points["Goals Against"].rolling(10).mean()

home_matches["Home Form"] = home_form
home_matches["Home Attack Form"] = home_attack_form
home_matches["Home Defense Form"] = home_defense_form

away_matches["Away Form"] = away_form
away_matches["Away Attack Form"] = away_attack_form
away_matches["Away Defense Form"] = away_defense_form

home_matches = home_matches[["Match ID","Season","Team","Home Form","Home Attack Form","Home Defense Form","Result"]]
home_matches = home_matches.rename(columns={"Team": "Home Team"})

away_matches = away_matches[["Match ID","Season","Team","Away Form","Away Attack Form","Away Defense Form","Result"]]
away_matches = away_matches.rename(columns={"Team": "Away Team"})

form_guide = home_matches.merge(away_matches, how='inner', on=['Match ID','Season','Result']).sort_values("Match ID")
form_guide = form_guide.reset_index(drop=True)

# Dropping the NaN values in home and away form
form_guide = form_guide.dropna(subset=["Home Form","Home Attack Form","Home Defense Form","Away Form","Away Attack Form","Away Defense Form"])
form_guide = form_guide.sort_values("Match ID")
form_guide = form_guide.reset_index(drop=True)

# Initialising ELO
elo = {}
home_elos = []
away_elos = []

# ELO constant initialisation
K = 20

# Loop through all the matches and update the elo dictionary after each match
for i in df.index:
    home_team = df.iloc[i]["home_team"]
    away_team = df.iloc[i]["away_team"]
    result   = df.iloc[i]["result"]
    season   = df.iloc[i]["season"]

    if home_team in elo:
        home_elo = elo[home_team]
    elif season == "2006-2007":
        home_elo = 1500
    else:
        home_elo = 1400

    if away_team in elo:
        away_elo = elo[away_team]
    elif season == "2006-2007":
        away_elo = 1500
    else:
        away_elo = 1400

    home_elos.append(home_elo)
    away_elos.append(away_elo)

    h_prob = 1 / (1 + 10**((away_elo - home_elo)/400))
    a_prob = 1 - h_prob

    if result == 'H':
        elo[home_team] = home_elo + K * (1 - h_prob)
        elo[away_team] = away_elo - K * (1 - h_prob)
    elif result == 'A':
        elo[home_team] = home_elo - K * (1 - a_prob)
        elo[away_team] = away_elo + K * (1 - a_prob)
    elif result == 'D':
        elo[home_team] = home_elo + K * (0.5 - h_prob)
        elo[away_team] = away_elo + K * (0.5 - a_prob)


df["Home ELO"] = home_elos
df["Away ELO"] = away_elos
print(df.iloc[500:510])






