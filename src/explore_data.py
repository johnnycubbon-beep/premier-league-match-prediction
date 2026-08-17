import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/raw/results.csv")
df["Match ID"] = range(len(df))

## Getting used to the syntax of pandas with common commands

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

## Creating a new, simpler and more readable DataFrame

# home_matches = df["home_team"].value_counts()
# home_wins = df[df["result"] == 'H']["home_team"].value_counts()

# team_home = pd.DataFrame({
#     "Home Matches": home_matches,
#     "Home Wins": home_wins
# })

# team_home["Home Win %"] = 100 * (team_home["Home Wins"]/team_home["Home Matches"])

# print(team_home.sort_values("Home Matches", ascending=False))
# print(type(team_home))
# print(type(df))

# away_matches = df["away_team"].value_counts()
# away_wins = df[df["result"] == 'A']["away_team"].value_counts()

# team_away = pd.DataFrame({
#     "Away Matches": away_matches,
#     "Away Wins": away_wins
#     })
# team_away["Away Win %"] = 100* (team_away["Away Wins"]/team_away["Away Matches"])

# print(team_away.sort_values("Away Win %", ascending=False))

# team_stats = team_home.join(team_away)
# print(team_stats.sort_values("Home Win %",ascending=False))

# team_stats["Total Matches"] = team_stats["Home Matches"] + team_stats["Away Matches"]
# team_stats["Total Wins"] = team_stats["Home Wins"] + team_stats["Away Wins"]
# team_stats["Win %"] = 100 * (team_stats["Total Wins"]/team_stats["Total Matches"])
# team_stats["Home-Away Diff"] = team_stats["Home Win %"] - team_stats["Away Win %"]
# print(team_stats.sort_values("Home-Away Diff", ascending=False)[["Home Win %", "Away Win %", "Win %","Home-Away Diff"]])
# teams_to_label = ["Liverpool", "Aston Villa", "Everton", "Chelsea", "Newcastle United","Fulham"]
# plt.scatter(team_stats[team_stats["Win %"] < 25]["Home Win %"], team_stats[team_stats["Win %"] < 25]["Away Win %"], s=20)
# plt.plot([0,100], [0,100], linestyle = '--')
# plt.xlabel("Home Win %")
# plt.ylabel("Away Win %")
# plt.title("Premier League Teams: Home vs Away Win %: 2006-07 - 2017-18")
# for team in team_stats[team_stats["Win %"] < 25].index:
#     plt.annotate(team, 
#                 (team_stats.loc[team, "Home Win %"], team_stats.loc[team, "Away Win %"]),
#                 xytext = (4,4),
#                 textcoords="offset points",
#                 fontsize = 4)
# plt.show()
# print(team_stats.loc[teams_to_label, "Home Win %"])

# season_results = df.groupby("season")["result"].value_counts(normalize=True)
# season_results = season_results.unstack()
# print(season_results[season_results['H'] > 0.47])

# season_results.plot()
# plt.xlabel("Season")
# plt.ylabel("Proportion of Matches")
# plt.title("Prem Match Results by Season")
# plt.show()++

# print(df.columns)

## Looking at goals and goal differences for each team across the 12 year period

# total_home_goals = df.groupby("home_team")["home_goals"].sum()
# total_away_goals = df.groupby("away_team")["away_goals"].sum()

# total_home_matches = df["home_team"].value_counts()
# total_away_matches = df["away_team"].value_counts()

# team_goals = pd.DataFrame({"Home For": total_home_goals, "Away For": total_away_goals, 
#                            "Home Matches": total_home_matches, "Away Matches": total_away_matches})

# team_goals["Home GPG"] = team_goals["Home For"]/team_goals["Home Matches"]
# team_goals["Away GPG"] = team_goals["Away For"]/team_goals["Away Matches"]
# team_goals["Total GPG"] = 0.5 * (team_goals["Home GPG"] + team_goals["Away GPG"])

# team_goals["Home Against"] = df.groupby("home_team")["away_goals"].sum()
# team_goals["Away Against"] = df.groupby("away_team")["home_goals"].sum()

# team_goals["Home GDPG"] = (team_goals["Home For"] - team_goals["Home Against"]) / team_goals["Home Matches"]
# team_goals["Away GDPG"] = (team_goals["Away For"] - team_goals["Away Against"]) / team_goals["Away Matches"]
# team_goals["Total GDPG"] = 0.5 * (team_goals["Home GDPG"] + team_goals["Away GDPG"])

# goal_diff = team_goals[["Home GDPG", "Away GDPG", "Total GDPG"]]
# print(goal_diff.sort_values("Total GDPG", ascending=False))

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

all_matches = pd.concat([home, away], ignore_index=True)
all_matches["Points"] = all_matches["Outcome"].map({'W': 3, 'D': 1, 'L': 0})
all_matches = all_matches.sort_values(["Team","Match ID"])



previous_points = all_matches.groupby("Team")["Points"].shift(1)
form = previous_points.rolling(5).mean()
all_matches["Form_5"] = form

# print(all_matches[all_matches["Team"] == "Liverpool"][40:50])

# home_points = all_matches[all_matches["Venue"]=="Home"].groupby("Team")["Points"].sum()
# print(home_points)

# high_goal_teams = all_matches[all_matches["Goals For"] > 6]["Team"].value_counts()
# print(high_goal_teams)

# boring_team = all_matches[(all_matches["Goals For"] == 0) & (all_matches["Goals Against"] == 0)]["Team"].value_counts()
# print(boring_team)

home_matches = all_matches[all_matches["Venue"]=="Home"].sort_values(["Team","Match ID"])
away_matches = all_matches[all_matches["Venue"]=="Away"].sort_values(["Team","Match ID"])

previous_home_points = home_matches["Points"].shift(1)
home_form = previous_home_points.rolling(5).mean()

previous_away_points = away_matches["Points"].shift(1)
away_form = previous_away_points.rolling(5).mean()

home_matches["Home Form"] = home_form
away_matches["Away Form"] = away_form

home_matches = home_matches[["Match ID", "Season","Team","Home Form"]]
home_matches = home_matches.rename(columns={"Team": "Home Team"})

away_matches = away_matches[["Match ID", "Season","Team","Away Form"]]
away_matches = away_matches.rename(columns={"Team": "Away Team"})

form_guide = home_matches.merge(away_matches, how='inner', on=['Match ID','Season']).sort_values("Match ID")
print(form_guide[2000:2020])

