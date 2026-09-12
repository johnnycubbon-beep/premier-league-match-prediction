import pandas as pd
import statsmodels.api as sm
import numpy as np
import json
import matplotlib.pyplot as plt



form_guide = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/prem_form_guide.csv")
df = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/full_prem_data.csv")

with open("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/relegated.json", "r") as f:
    relegated_teams = json.load(f)
with open("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/promoted.json", "r") as f:
    promoted_teams = json.load(f)


# Getting the match IDs for each of the opening games of the season
season_start_indices = df.groupby("season").first()["Match ID"].iloc[1:].values

# Initialising ELO
elo = {}
home_elos = []
away_elos = []

# ELO constant initialisations. K determines the size of the changes after each match. 
# d determines how much promoted team ELO is discounted.
K = 20
d = 150

# Loop through all the matches and update the elo dictionary after each match
for i in df.index:
    
    home_team = df.iloc[i]["home_team"]
    away_team = df.iloc[i]["away_team"]
    result   = df.iloc[i]["result"]
    season   = df.iloc[i]["season"]

    if i in season_start_indices:
        last_season = df.iloc[i-1]["season"]
        
        for team in relegated_teams[last_season]:
            elo.pop(team)

    # If team not seen before initialise elo as 1500
    if home_team in elo:
        home_elo = elo[home_team]
    elif season == "1993-1994":
        home_elo = 1500
    else:
        home_elo = (sum(elo.values())/len(elo.values())) - d

    if away_team in elo:
        away_elo = elo[away_team]
    elif season == "1993-1994":
        away_elo = 1500
    else:
        away_elo = (sum(elo.values())/len(elo.values())) - d

    # Add to respective elo lists
    home_elos.append(home_elo)
    away_elos.append(away_elo)

    # Home win and away win probabilities
    h_prob = 1 / (1 + 10**((away_elo - home_elo)/400))
    a_prob = 1 - h_prob

    # Getting the home score
    if result == 'H':
        h_score = 1
    elif result == 'A':
        h_score = 0
    elif result == 'D':
        h_score = 0.5
    a_score = 1 - h_score

    # Formula for the ELO changes: if h_prob close to zero and h_score==1 biggest change
    elo[home_team] = home_elo + K * (h_score - h_prob)
    elo[away_team] = away_elo + K * (a_score - a_prob)

# Adding ELO columns to both df and form_guide 
df["Home ELO"] = home_elos
df["Away ELO"] = away_elos
elo_data = df[["Match ID","Home ELO","Away ELO","home_goals","away_goals"]]
form_guide = form_guide.merge(elo_data,on="Match ID",how="left")

# Adding ELO difference and form difference to form guide
form_guide["ELO diff"] = form_guide["Home ELO"] - form_guide["Away ELO"]
form_guide["Form diff"] = form_guide["Home Form"] - form_guide["Away Form"]
form_guide["Attack diff"] = form_guide["Home Attack Form"] - form_guide["Away Attack Form"]
form_guide["Defense diff"] = form_guide["Home Defense Form"] - form_guide["Away Defense Form"]

# Define Predictors and Response Variables
X = form_guide[[
    "ELO diff",
    "Form diff"
    ]]

# Add column of ones in the predictor matrix
X = sm.add_constant(X)
Y = form_guide["home_goals"]

# Define GLM model
model = sm.GLM(
    Y,
    X,
    family=sm.families.Poisson()
)
# Fit the model
results = model.fit()

# Get model parameters
print(results.params)

new_match = pd.DataFrame({
    "const": [1],
    "ELO diff":[-200],
    "Form diff":[1]
})

print(results.predict(new_match))