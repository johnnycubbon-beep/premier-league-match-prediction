"""Train and evaluate a random forest on the prepared match features."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "processed" / "prem_form_guide.csv"

form_guide = pd.read_csv(DATA_FILE)
X = form_guide[["Home ELO", "Away ELO", "Home Form", "Away Form"]]
y = form_guide["Result"]

train_end = form_guide[form_guide["Season"] == "2020-2021"].index[0]
test_end = form_guide[form_guide["Season"] == "2025-2026"].index[0]
X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=6,
    max_features=3,
    bootstrap=True,
)
forest.fit(X_train, y_train)

train_accuracy = forest.score(X_train, y_train)
test_accuracy = forest.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.2%}")
print(f"Test accuracy (2020-2025): {test_accuracy:.2%}")

ConfusionMatrixDisplay.from_predictions(y_test, forest.predict(X_test), display_labels=forest.classes_)
plt.tight_layout()
plt.show()
