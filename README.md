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

The project also explores rating-based approaches such as Elo ratings.

An important consideration throughout the project is avoiding temporal leakage: features for a match must only use information that would have been available immediately before that match.

## Models

The project investigates several approaches, including:

- Decision trees
- Random forests
- Elo-based models
- Simple baseline predictors

Decision-tree complexity is varied to investigate the bias-variance trade-off and the effect of overfitting.

For example, increasingly deep trees can fit the training data extremely well while performing substantially worse on unseen matches.

## Results

Simple recent-form features provide only modest predictive power.

For example, using recent goal-form features with a small decision tree produced test accuracy around:

**53%**

This is only a modest improvement over simple baselines, highlighting how difficult football prediction is.

Increasing tree depth can dramatically increase training performance without improving — and often while worsening — test performance.

This provides a useful practical illustration of the bias-variance trade-off.

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

The project has also motivated further investigation into probabilistic models, including Poisson models for football scores.

## Project structure

```text
prem_prediction/
├── data/
├── notebooks/
├── src/
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── elo.py
├── results/
├── README.md
└── requirements.txt
