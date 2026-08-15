import pandas as pd
df = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/raw/results.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df["result"].value_counts())
# print(df["season"].value_counts())
# print(df["result"].value_counts(normalize=True) * 100)
# print(type(df.groupby("season")))
# print(type(df["result"]))
# print(type(df))
# print(type(df.groupby("season")["result"]))
# print(df.groupby("season")["result"].value_counts(normalize=True)*100)
# print(df["home_team"].unique())
# print(df["home_team"].nunique())
# print(df[df["result"] == 'H']["home_team"].value_counts().shape)
# print(df["home_team"].value_counts().shape)
# print(100 * df[df["result"] == 'H']["home_team"].value_counts() / df["home_team"].value_counts() )
# print(df[df["result"] == 'A']["away_team"].value_counts())
home_matches = df["home_team"].value_counts()
home_wins = df[df["result"] == 'H']["home_team"].value_counts()

team_home = pd.DataFrame({
    "Home Matches": home_matches,
    "Home Wins": home_wins
})

team_home["Home Win %"] = 100 * (team_home["Home Wins"]/team_home["Home Matches"])

# print(team_home.sort_values("Home Matches", ascending=False))
# print(type(team_home))
# print(type(df))

away_matches = df["away_team"].value_counts()
away_wins = df[df["result"] == 'A']["away_team"].value_counts()

team_away = pd.DataFrame({
    "Away Matches": away_matches,
    "Away Wins": away_wins
    })
team_away["Away Win %"] = 100* (team_away["Away Wins"]/team_away["Away Matches"])

# print(team_away.sort_values("Away Win %", ascending=False))

team_stats = team_home.join(team_away)
# print(team_stats.sort_values("Home Win %",ascending=False))

team_stats["Total Matches"] = team_stats["Home Matches"] + team_stats["Away Matches"]
team_stats["Total Wins"] = team_stats["Home Wins"] + team_stats["Away Wins"]
team_stats["Win %"] = 100 * (team_stats["Total Wins"]/team_stats["Total Matches"])
team_stats["Home - Away Diff"] = team_stats["Home Win %"] - team_stats["Away Win %"]
print(team_stats.sort_values("Home - Away Diff", ascending=False)["Home - Away Diff"])

