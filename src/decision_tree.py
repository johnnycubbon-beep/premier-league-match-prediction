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

# print(df[(df["away_team"] == "Arsenal") & (1900 < df["Match ID"]) & (df["Match ID"] < 2040)][["Match ID","home_goals"]])
# print(form_guide[(1900 < form_guide["Match ID"]) & (form_guide["Match ID"] < 2040) & (form_guide["Away Team"] == "Arsenal")][["Match ID","Away Defense Form"]])


# Initialise Training and Test Data
split = int(0.8 * len(form_guide))
form_guide["Basic Pred"] = form_guide["Home Form"] > form_guide["Away Form"]
form_guide["Basic Pred"] = form_guide["Basic Pred"].map({True: "H", False: "A"})
form_guide["Correct"] = form_guide["Basic Pred"] == form_guide["Result"]
form_guide["Correct"] = form_guide["Correct"].map({True: 1, False: 0})




# Define Predictors and Response Variables
X = form_guide[["Home Form", 
                "Away Form",
                "Home Attack Form",
                "Home Defense Form",
                "Away Attack Form",
                "Away Defense Form"]]
Y = form_guide["Result"]





# Initialise Training and Test Data
split = int(0.8 * len(form_guide))

# Predictor Data
X_train = X.iloc[:split]
X_test = X.iloc[split:]

# Response Data
Y_train = Y.iloc[:split]
Y_test = Y.iloc[split:]

# print(X_train.shape)
# print(X_test.shape)

# Run decision tree from scikit learn
tree_goals = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

# Fit to the training data
tree_goals.fit(X_train, Y_train)

# Make Predictions
Y_pred = tree_goals.predict(X_test)

# Compare to the actual results and look at accuracy
accuracy = accuracy_score(Y_test, Y_pred)

# Get training and test accuracy for our new model with extra predictors
print(tree_goals.score(X_train,Y_train))
print(tree_goals.score(X_test,Y_test))

# # Looking at what the tree actually looks like
# print(tree.get_depth())
# print(tree.get_n_leaves)

# # Seeing whether there are differences in indentifying home vs away wins
# print(confusion_matrix(Y_test, Y_pred, labels=["H", "D", "A"]))

# # Plotting the tree
# plt.figure(figsize=(15,8))

# plot_tree(
#     tree,
#     feature_names=["Home Form","Away Form"],
#     class_names=["A", "D", "H"],
#     filled=True
# )

# plt.show()

# # Bootstrapping to see if the tree is genuinely better than the simple method
B = 10000
vals = []
acc_simple = form_guide["Correct"][split:]
acc_tree = (Y_test == Y_pred).map({True: 1, False: 0})
est = acc_tree.mean() - acc_simple.mean()
for _ in range(B):
    indices = np.random.randint(0, len(Y_test), size=len(Y_test))
    simple = acc_simple.iloc[indices].mean()
    tree = acc_tree.iloc[indices].mean()
    boot_est = tree - simple
    vals.append(np.sqrt(len(acc_tree)) * (boot_est - est))

quantiles = np.percentile(vals, [2.5, 97.5])
conf_int = [0,0]
conf_int[0] = est - quantiles[1] / (np.sqrt(len(acc_tree)))
conf_int[1] = est - quantiles[0] / (np.sqrt(len(acc_tree)))
print(conf_int)
# Testing on multiple different tree depths (ie changing the complexity of the hypothesis class)
# depths = [1,2,3,4,5,6,8,10]
# train_accuracy = []
# test_accuracy = []

# for depth in depths:
#     tree = DecisionTreeClassifier(
#         criterion="gini",
#         max_depth=depth,
#         random_state=42
#     )

#     tree.fit(X_train, Y_train)
#     train_accuracy.append(tree.score(X_train,Y_train))
#     test_accuracy.append(tree.score(X_test,Y_test))

# print(train_accuracy)
# print(test_accuracy)

# # Trying an unresticted tree since at depth 10 the training accuracy isn't even that high
# tree_full = DecisionTreeClassifier(
#     criterion="gini",
#     random_state=42
# )
# tree_full.fit(X_train, Y_train)

# print(tree_full.score(X_train, Y_train))
# print(tree_full.score(X_test, Y_test))
# print(tree_full.get_depth())
# print(tree_full.get_n_leaves())

# Looking at whether any of the predictor values are duplicated
# dup_array = X_train.duplicated()
# print(len(X_train))
# print(len(X_train[dup_array]))
# print(X_train.value_counts())

