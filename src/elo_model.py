import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import numpy as np

seasons = [f"{i}-{i+1}" for i in range(1993,2026)]
dfs = []

for season in seasons:
    path = "C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/raw/" + season + ".csv"
    try:
        df = pd.read_csv(
            path,
            usecols=["HomeTeam","AwayTeam","FTHG","FTAG","FTR"],
            encoding="latin1")
        df["season"] = season
        dfs.append(df)
    except Exception as e:
        print(f"ERROR in season {season}: {e}")
        break
df = pd.concat(dfs, ignore_index=True)


# Setting column with match ID helps to get around indices getting messed up
df["Match ID"] = range(len(df))

# Changing the names of the columns so that they match the original results.csv
df = df.rename(columns = {"HomeTeam":"home_team",
                     "AwayTeam":"away_team",
                     "FTHG":"home_goals",
                     "FTAG":"away_goals",
                     "FTR":"result"})




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

# Home rolling means for goals for and against and points
previous_home_points = home_matches.groupby("Team")[["Points","Goals For","Goals Against"]].shift(1)
home_form = previous_home_points["Points"].rolling(6).mean()
home_attack_form = previous_home_points["Goals For"].rolling(10).mean()
home_defense_form = previous_home_points["Goals Against"].rolling(10).mean()


# Away rolling means for goals for and against and points
previous_away_points = away_matches.groupby("Team")[["Points","Goals For","Goals Against"]].shift(1)
away_form = previous_away_points["Points"].rolling(6).mean()
away_attack_form = previous_away_points["Goals For"].rolling(10).mean()
away_defense_form = previous_away_points["Goals Against"].rolling(10).mean()

# Adding form metrics to home_matches
home_matches["Home Form"] = home_form
home_matches["Home Attack Form"] = home_attack_form
home_matches["Home Defense Form"] = home_defense_form

# Adding form metrics to away matches
away_matches["Away Form"] = away_form
away_matches["Away Attack Form"] = away_attack_form
away_matches["Away Defense Form"] = away_defense_form

home_matches = home_matches[["Match ID","Season","Team","Home Form","Home Attack Form","Home Defense Form","Result"]]
home_matches = home_matches.rename(columns={"Team": "Home Team"})

away_matches = away_matches[["Match ID","Season","Team","Away Form","Away Attack Form","Away Defense Form","Result"]]
away_matches = away_matches.rename(columns={"Team": "Away Team"})

# Merging home and away dataframes according to Match ID, Season and Result
form_guide = home_matches.merge(away_matches, how='inner', on=['Match ID','Season','Result']).sort_values("Match ID")
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

    # If team not seen before initialise elo as 1500
    if home_team in elo:
        home_elo = elo[home_team]
    else:
        home_elo = 1500

    if away_team in elo:
        away_elo = elo[away_team]
    else:
        away_elo = 1500

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
form_guide["Home ELO"] = home_elos
form_guide["Away ELO"] = away_elos

# Dropping the NaN values in home and away form
form_guide = form_guide.dropna(subset=["Home Form","Home Attack Form","Home Defense Form","Away Form","Away Attack Form","Away Defense Form"])
form_guide = form_guide.sort_values("Match ID")
form_guide = form_guide.reset_index(drop=True)

# Basic Prediction Method
form_guide["Basic Pred"] = form_guide["Home Form"] > form_guide["Away Form"]
form_guide["Basic Pred"] = form_guide["Basic Pred"].map({True: "H", False: "A"})
form_guide["Correct"] = form_guide["Basic Pred"] == form_guide["Result"]
form_guide["Correct"] = form_guide["Correct"].map({True: 1, False: 0})


# Define Predictors and Response Variables
X = form_guide[["Home Form",
                "Away Form",
                "Home ELO",
                "Away ELO"]]
Y = form_guide["Result"]

# Choose training to test split at the start of the 2020-2021 season (Liverpool Champions)
split_0 = form_guide[form_guide["Season"] == "2006-2007"].index[0]
split_1 = form_guide[form_guide["Season"] == "2015-2016"].index[-78]
split_2 = form_guide[form_guide["Season"] == "2017-2018"].index[-1]

# print(form_guide.iloc[split_1:split_2]["Season"].value_counts())

# Predictor Data
X_train = X.iloc[split_0:split_1]
X_test = X.iloc[split_1:split_2]

# Response Data
Y_train = Y.iloc[split_0:split_1]
Y_test = Y.iloc[split_1:split_2]


# Run decision tree from scikit learn
tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42)

# Fit to the training data
tree.fit(X_train, Y_train)

# Make Predictions on the test data
Y_pred = tree.predict(X_test)

# Get the accuracy on the training and test data
print(tree.score(X_train,Y_train))
print(tree.score(X_test,Y_test))

# # Bootstrapping to see if the tree is genuinely better than the simple method
# B = 10000
# vals = []
# vals_tree = []
# acc_simple = form_guide["Correct"][split:]
# print(f"Mean Accuracy for the Simple Method of Prediction is {acc_simple.mean()}")
# acc_tree = (Y_test == Y_pred).map({True: 1, False: 0})
# print(f"Mean Accuracy for the Decision Tree is {acc_tree.mean()}")
# est = acc_tree.mean() - acc_simple.mean()
# for _ in range(B):
#     indices = np.random.randint(0, len(Y_test), size=len(Y_test))
#     simple = acc_simple.iloc[indices].mean()
#     tree = acc_tree.iloc[indices].mean()
#     boot_est = tree - simple
#     vals.append(np.sqrt(len(acc_tree)) * (boot_est - est))
#     vals_tree.append(simple)

# quantiles = np.percentile(vals, [2.5, 97.5])
# quantiles_2 = np.percentile(vals_tree,[2.5, 97.5])
# print(quantiles_2)
# conf_int = [0,0]
# conf_int[0] = est - quantiles[1] / (np.sqrt(len(acc_tree)))
# conf_int[1] = est - quantiles[0] / (np.sqrt(len(acc_tree)))
# print(conf_int)






