import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import numpy as np
import json

seasons = [f"{i}-{i+1}" for i in range(1993,2026)] 
L = len(seasons)
# dfs = []
# for season in seasons:
#     path = "C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/raw/" + season + ".csv"
#     try:
#         df = pd.read_csv(
#             path,
#             usecols=["HomeTeam","AwayTeam","FTHG","FTAG","FTR"],
#             encoding="latin1")
#         df["season"] = season
#         dfs.append(df)
#     except Exception as e:
#         print(f"ERROR in season {season}: {e}")
#         break
# df = pd.concat(dfs, ignore_index=True)
# df = df.dropna()



# # Setting column with match ID helps to get around indices getting messed up
# df["Match ID"] = range(len(df))

# # Getting the 20 teams from each season
# season_teams = df.groupby("season")["HomeTeam"].unique()

# # Creating a dictionary with the keys as the seasons and the values as the list of teams relegated
# promoted_teams = {}
# for i in range(L-1):
#     prom = []
#     season_now = seasons[i]
#     season_next = seasons[i+1]
#     teams_now = season_teams.loc[season_now]
#     teams_next = season_teams.loc[season_next]

#     for team in teams_next:
#         if team not in teams_now:
#             prom.append(team)
#     promoted_teams[season_now] = prom

# with open("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/promoted.json", "w") as f:
#     json.dump(promoted_teams, f, indent=4)





# # Changing the names of the columns so that they match the original results.csv
# df = df.rename(columns = {"HomeTeam":"home_team",
#                      "AwayTeam":"away_team",
#                      "FTHG":"home_goals",
#                      "FTAG":"away_goals",
#                      "FTR":"result"})

# df.to_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/full_prem_data.csv", index=False)




# # Gather all the home games for each team so change result column to Win, Lose or Draw
# # Add a Venue column to keep track once we merge Home and Away games for each team
# home = df[["Match ID","season", "home_team", "home_goals", "away_goals", "result"]].copy()
# home = home.rename(columns={"season":"Season",
#                             "home_team": "Team",
#                             "home_goals": "Goals For",
#                             "away_goals": "Goals Against",
#                             "result": "Result"
#                             })
# home["Outcome"] = home["Result"].map({
#     "H": "W",
#     "D": "D",
#     "A": "L"
# })
# home["Venue"] = "Home"

# # Gather all the away games for each team so change result column to Win, Lose or Draw
# # Add a Venue column to keep track once we merge Home and Away games for each team
# away = df[["Match ID","season","away_team","home_goals", "away_goals", "result"]].copy()
# away = away.rename(columns={"season": "Season",
#                             "away_team": "Team",
#                             "home_goals": "Goals Against",
#                             "away_goals": "Goals For",
#                             "result": "Result"
#                             })
# away["Outcome"] = away["Result"].map({
#     'H':'L',
#     'D':'D',
#     'A':'W'
#     })
# away["Venue"] = "Away"

# # Concatenate home and away frames so that we now have double the rows and two rows corresponding to each match
# # Map Win Lose Draw to the points values of each
# all_matches = pd.concat([home, away], ignore_index=True)
# all_matches["Points"] = all_matches["Outcome"].map({'W': 3, 'D': 1, 'L': 0})
# all_matches = all_matches.sort_values(["Team","Match ID"])

# # Re split the home and away matches (last code was somewhat unnecessary)
# home_matches = all_matches[all_matches["Venue"]=="Home"].sort_values(["Team","Match ID"])
# away_matches = all_matches[all_matches["Venue"]=="Away"].sort_values(["Team","Match ID"])

# # Home rolling means for goals for and against and points
# home_form = (
#     home_matches.groupby("Team")["Points"]
#     .transform(lambda x: x.shift(1).rolling(6).mean())
# )

# home_attack_form = (
#     home_matches.groupby("Team")["Goals For"]
#     .transform(lambda x: x.shift(1).rolling(10).mean())
# )

# home_defense_form = (
#     home_matches.groupby("Team")["Goals Against"]
#     .transform(lambda x: x.shift(1).rolling(10).mean())
# )


# # Away rolling means for goals for and against and points
# away_form = (
#     away_matches.groupby("Team")["Points"]
#     .transform(lambda x: x.shift(1).rolling(6).mean())
# )

# away_attack_form = (
#     away_matches.groupby("Team")["Goals For"]
#     .transform(lambda x: x.shift(1).rolling(10).mean())
# )

# away_defense_form = (
#     away_matches.groupby("Team")["Goals Against"]
#     .transform(lambda x: x.shift(1).rolling(10).mean())
# )


# # Adding form metrics to home_matches
# home_matches["Home Form"] = home_form
# home_matches["Home Attack Form"] = home_attack_form
# home_matches["Home Defense Form"] = home_defense_form



# # Adding form metrics to away matches
# away_matches["Away Form"] = away_form
# away_matches["Away Attack Form"] = away_attack_form
# away_matches["Away Defense Form"] = away_defense_form

# home_matches = home_matches[["Match ID","Season","Team","Home Form","Home Attack Form","Home Defense Form","Result"]]
# home_matches = home_matches.rename(columns={"Team": "Home Team"})

# away_matches = away_matches[["Match ID","Season","Team","Away Form","Away Attack Form","Away Defense Form","Result"]]
# away_matches = away_matches.rename(columns={"Team": "Away Team"})

# # Merging home and away dataframes according to Match ID, Season and Result
# form_guide = home_matches.merge(away_matches, how='inner', on=['Match ID','Season','Result']).sort_values("Match ID")
# form_guide = form_guide.reset_index(drop=True)

# # Dropping the NaN values in home and away form
# form_guide = form_guide.dropna(subset=["Home Form","Home Attack Form","Home Defense Form","Away Form","Away Attack Form","Away Defense Form"])
# form_guide = form_guide.sort_values("Match ID")
# form_guide = form_guide.reset_index(drop=True)

# # Basic Prediction Method
# form_guide["Basic Pred"] = form_guide["Home Form"] > form_guide["Away Form"]
# form_guide["Basic Pred"] = form_guide["Basic Pred"].map({True: "H", False: "A"})
# form_guide["Correct"] = form_guide["Basic Pred"] == form_guide["Result"]
# form_guide["Correct"] = form_guide["Correct"].map({True: 1, False: 0})

# form_guide.to_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/prem_form_guide.csv")


form_guide = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/prem_form_guide.csv")
df = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/full_prem_data.csv")

with open("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/relegated.json", "r") as f:
    relegated_teams = json.load(f)
with open("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/promoted.json", "r") as f:
    promoted_teams = json.load(f)



season_start_indices = df.groupby("season").first()["Match ID"].iloc[1:].values

# print(form_guide.shape)
# print(df.shape)
# print(relegated_teams)
# print(season_start_indices)
# print(promoted_teams)




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
elo_data = df[["Match ID","Home ELO","Away ELO"]]
form_guide = form_guide.merge(elo_data,on="Match ID",how="left")


# print(df[df["Match ID"]==11003][["home_team","away_team","Home ELO","Away ELO"]])
# print(form_guide[form_guide["Match ID"]==11003][["Home Team","Away Team","Home ELO","Away ELO"]])




#Define Predictors and Response Variables
X = form_guide[[
    "Home ELO",
    "Away ELO",
    "Home Form",
    "Away Form"]]
Y = form_guide["Result"]

# Keeping 2020-2021 to 2025-2026 as precious test data then tuning the parameters 
# using everything up to this point 
split_1 = form_guide[form_guide["Season"] == "2016-2017"].index[0]
split_2 = form_guide[form_guide["Season"] == "2018-2019"].index[0]

# Predictor Data
X_train = X.iloc[:split_1]
X_test = X.iloc[split_1:split_2]

# Response Data
Y_train = Y.iloc[:split_1]
Y_test = Y.iloc[split_1:split_2]

# Initialise the random forest
forest = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=6,
    max_features=3,
    bootstrap=True,
    max_samples=None
)
# Fit the forest to our training data
forest.fit(X_train,Y_train)

print(forest.n_classes_)
# Make predictions on the test data
Y_pred = forest.predict(X_test)

# Get the training and test accuracy of the fitted estimator
train_acc = forest.score(X_train,Y_train)
test_acc = forest.score(X_test,Y_test)
print(f"Training Accuracy for the model is {train_acc*100}%. Test Accuracy is {test_acc*100}%.")

# pred = [res == "H" for res in Y_pred]
# test = [res == "H" for res in Y_test]
# arr = np.stack([pred,test],axis=1)
# correct = [(pair[0] == pair[1]) for pair in arr]
# acc = sum(correct)/len(correct)
# print(acc)
