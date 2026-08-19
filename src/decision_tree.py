import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

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

previous_home_points = home_matches.groupby("Team")["Points"].shift(1)
home_form = previous_home_points.rolling(5).mean()



previous_away_points = away_matches.groupby("Team")["Points"].shift(1)
away_form = previous_away_points.rolling(5).mean()

home_matches["Home Form"] = home_form
away_matches["Away Form"] = away_form

home_matches = home_matches[["Match ID", "Season","Team","Home Form","Result"]]
home_matches = home_matches.rename(columns={"Team": "Home Team"})

away_matches = away_matches[["Match ID", "Season","Team","Away Form","Result"]]
away_matches = away_matches.rename(columns={"Team": "Away Team"})

form_guide = home_matches.merge(away_matches, how='inner', on=['Match ID','Season','Result']).sort_values("Match ID")
form_guide = form_guide.reset_index(drop=True)

# Dropping the NaN values in home and away form
form_guide = form_guide.dropna(subset=["Home Form", "Away Form"])
form_guide = form_guide.sort_values("Match ID")
form_guide = form_guide.reset_index(drop=True)

# Define Predictors and Response Variables
X = form_guide[["Home Form", "Away Form"]]
Y = form_guide["Result"]


# Initialise Training and Test Data
split = int(0.8 * len(form_guide))

# Predictor Data
X_train = X.iloc[:split]
X_test = X.iloc[split:]

# Response Data
Y_train = Y.iloc[:split]
Y_test = Y.iloc[split:]

# Run decision tree from scikit learn
tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

# Fit to the training data
tree.fit(X_train, Y_train)

# Make Predictions
Y_pred = tree.predict(X_test)

# Compare to the actual results and look at accuracy
accuracy = accuracy_score(Y_test, Y_pred)

# Looking at what the tree actually looks like
print(tree.get_depth())
print(tree.get_n_leaves)

# Seeing whether there are differences in indentifying home vs away wins
print(confusion_matrix(Y_test, Y_pred, labels=["H", "D", "A"]))

# Plotting the tree
plt.figure(figsize=(15,8))

plot_tree(
    tree,
    feature_names=["Home Form","Away Form"],
    class_names=["A", "D", "H"],
    filled=True
)

plt.show()










