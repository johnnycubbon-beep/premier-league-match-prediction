import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import json
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score


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
form_guide["ELO_diff"] = form_guide["Home ELO"] - form_guide["Away ELO"]
form_guide["Form_diff"] = form_guide["Home Form"] - form_guide["Away Form"]
form_guide["Attack_diff"] = form_guide["Home Attack Form"] - form_guide["Away Attack Form"]
form_guide["Defense_diff"] = form_guide["Home Defense Form"] - form_guide["Away Defense Form"]

# print(form_guide.columns)

# Defining the training, validation and test data
train_split = form_guide[form_guide["Season"]=="2018-2019"].index[0]
val_split = form_guide[form_guide["Season"]=="2020-2021"].index[0]

train_data = form_guide.iloc[:train_split]
val_data = form_guide.iloc[train_split:val_split]
test_data = form_guide.iloc[val_split:]

# # FIRST IMPLEMENTATION OF THE GLM

# # Define Predictors and Response Variables
# X = train_data[[
#     "ELO_diff",
#     "Form_diff",
#     "Attack_diff",
#     "Defense_diff"
#     ]]

# # Add column of ones in the predictor matrix
# X = sm.add_constant(X)
# Y = train_data["home_goals"]

# # Define GLM model
# model = sm.GLM(
#     Y,
#     X,
#     family=sm.families.Poisson()
# )
# # Fit the model
# results = model.fit()

# # Get model parameters
# print(results.params)

# new_match = pd.DataFrame({
#     "const": [1],
#     "ELO_diff":[-200],
#     "Form_diff":[1]
# })

# print(results.predict(new_match))




# SECOND IMPLEMENTATION OF THE GLM
home_model = smf.glm(
    formula="home_goals ~ ELO_diff + Form_diff + Attack_diff + Defense_diff",
    data=train_data,
    family=sm.families.Poisson()
)
home_results = home_model.fit()
# print(home_results.summary())

away_model = smf.glm(
    formula="away_goals ~ ELO_diff + Form_diff + Attack_diff + Defense_diff",
    data=train_data,
    family=sm.families.Poisson()
)

away_results = away_model.fit()
# print(away_results.summary())





def get_probabilities(l_H,l_A):

    N = max(l_H,l_A)
    poisson_func = lambda x: (x[1]**x[0] * np.exp(-x[1]))/math.factorial(x[0])
    home_probs = [poisson_func([x,l_H]) for x in range(int(5*N))]
    away_probs = [poisson_func([x,l_A]) for x in range(int(5*N))]

    home_sum = 0
    draw_sum = 0
    away_sum = 0
    for i in range(int(5*N)):
        h_prob = home_probs[i]
        home_sum += (h_prob * sum(away_probs[:i]))
        draw_sum += (h_prob * away_probs[i])
        away_sum += (h_prob * sum(away_probs[i+1:]))

    return {"H":home_sum,"D":draw_sum,"A":away_sum}



predictions = []

for _, row in val_data.iterrows():
    X = pd.DataFrame({
        "ELO_diff": [row["ELO_diff"]],
        "Form_diff":[row["Form_diff"]],
        "Attack_diff":[row["Attack_diff"]],
        "Defense_diff":[row["Defense_diff"]]
    })
    lambda_h = home_results.predict(X).iloc[0]
    lambda_a = away_results.predict(X).iloc[0]

    p_h = get_probabilities(lambda_h,lambda_a)["H"]
    p_d = get_probabilities(lambda_h,lambda_a)["D"]
    p_a = get_probabilities(lambda_h,lambda_a)["A"]

    predictions.append(["H","D","A"][np.argmax([p_h,p_d,p_a])])

accuracy = accuracy_score(val_data["Result"],predictions)
print(accuracy)


