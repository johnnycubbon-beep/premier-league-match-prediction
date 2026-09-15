import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import json
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
import time
from scipy.stats import poisson

df = pd.read_csv("C:/Users/johnn/Documents/Python/Prem_Prediction_New/Data/processed/full_prem_data.csv")
df = df[df["season"].isin(["1993-1994","1994-1995","1995-1996"])]
home_probs = df["home_goals"].value_counts()/len(df)
away_probs = df["away_goals"].value_counts()/len(df)
result_probs = df[["home_goals","away_goals"]].value_counts()/len(df)
print(home_probs.index)
print(result_probs.index)
print(result_probs.loc[(2.0,1.0)])

indep_ratios = np.zeros((10,10))
for h in range(10):
    for a in range(10):
        try:
            indep_ratios[h,a] = result_probs.loc[(float(h),float(a))]/(home_probs.loc[float(h)] * away_probs.loc[float(a)])
        except:
            indep_ratios[h,a] = 0

for h,a in [(0,0), (1,0), (0,1), (1,1)]:
    print(f"{h}-{a}: "
          f"{indep_ratios[h,a]:.3f}")