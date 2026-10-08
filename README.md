<<<<<<< HEAD
# Premier League Match Prediction

A statistical and machine learning project investigating how well Premier League match outcomes can be predicted from historical match and team-performance data.

The project focuses not only on predictive accuracy, but also on understanding what information is actually useful for prediction and the limitations imposed by the inherently noisy nature of football.

## Project objectives

The main aims are to:

- Build predictive models for Premier League match outcomes.
- Construct features using information available before each match.
- Investigate whether recent team form contains useful predictive information.
- Compare different machine learning approaches.
- Examine overfitting, model complexity and out-of-sample performance.
- Understand how much of football match outcome variation is fundamentally unpredictable.

## Data

The initial dataset contains Premier League match results from the 2006/07 to 2017/18 seasons.

Each match contains information such as:

- Home team
- Away team
- Match result
- Goals scored by each team

The dataset contains 4,560 matches across 39 teams.

The response variable is the match outcome:

- `H` — home win
- `D` — draw
- `A` — away win

Features are constructed using historical information so that future match results do not leak into the predictors.

## Feature engineering

One of the first approaches investigated was a **form-based model**.

For each team, historical match results were used to construct recent-form features. Goal-based form was also investigated, including rolling averages over different historical windows.

The project also explores rating-based approaches such as Elo ratings. Elo ratings ended up having the most predictive power.

An important consideration throughout the project is avoiding temporal leakage: features for a match must only use information that would have been available immediately before that match.

## Models

The project investigates several approaches, including:

- Decision trees
- Random forests
- Elo-based models
- Poisson GLM 
- Simple baseline predictors

Decision-tree complexity is varied to investigate the bias-variance trade-off and the effect of overfitting.

For example, increasingly deep trees can fit the training data extremely well while performing substantially worse on unseen matches.

## Results

Simple recent-form features provide only modest predictive power.

Using ELO features and a Poisson GLM model, we can achieve prediction accuracy between 56% and 58%, outperforming both Decision Trees and Random Forests. Future improvements will involve comparing probabilities outputted by the GLM to bookies odds as well as additional feature engineering ("days since most recent game", "relative importance of the game" etc).



## Statistical considerations

Football outcomes contain a large amount of irreducible variation.

Even with perfect knowledge of historical team performance, the result of an individual match remains uncertain because of factors such as:

- Injuries
- Tactical decisions
- Individual mistakes
- Refereeing decisions
- Random variation in finishing
- Game-state effects
- Unobserved differences between teams

Consequently, high predictive accuracy should not be expected from a model predicting individual match outcomes.

The project therefore places emphasis on **out-of-sample evaluation** rather than training accuracy.

## Key lessons

Some of the most useful conclusions from the project have been methodological rather than purely predictive:

1. **More complex models do not necessarily predict better.**
2. **Feature construction is often more important than model complexity.**
3. **Temporal leakage is an important risk in sports prediction.**
4. **Training accuracy can give a misleading impression of model quality.**
5. **Football contains substantial irreducible uncertainty.**



=======
# Premier League Match Prediction

This project explores Premier League match outcomes with decision trees, a random forest, Elo ratings, and Poisson goal models.

## Data workflow

The season files in `Data/raw/` are the source match tables. Run the preparation script from any working directory:

```powershell
python src/prepare_data.py
```

The script writes the canonical processed match table and the pre-match feature table to `Data/processed/`. It calculates form from each team's previous home or away matches and stores pre-match Elo ratings. The model scripts read these shared feature files instead of rebuilding them.

Run an individual analysis from the project root:

```powershell
python src/decision_tree.py
python src/random_forest.py
python src/elo_model.py
python src/Poisson_GLM.py
python src/tune_elo_glm.py
python src/explore_data.py
python src/dixon_coles_check.py
```

The Elo tuning script compares K values, promoted-team starting ratings, and relegation handling on 2018–19 and 2019–20. It saves its leaderboard to `Output/elo_glm_tuning.csv` and leaves 2020–21 onward out of parameter selection.

Install the packages listed in `requirements.txt` with `pip install -r requirements.txt`.
>>>>>>> 6b66ed2 (Optimisation over Parameters K and d in ELO claculation.)
