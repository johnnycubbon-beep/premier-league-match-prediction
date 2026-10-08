"""Decision tree using the prepared Elo ratings and recent points form."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "processed" / "prem_form_guide.csv"

form_guide = pd.read_csv(DATA_FILE)
features = ["Home ELO", "Away ELO", "Home Form", "Away Form"]
X = form_guide[features]
y = form_guide["Result"]

train_end = form_guide[form_guide["Season"] == "2020-2021"].index[0]
test_end = form_guide[form_guide["Season"] == "2025-2026"].index[0]
X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

print(f"Training accuracy: {tree.score(X_train, y_train):.2%}")
print(f"Test accuracy (2020–2026): {tree.score(X_test, y_test):.2%}")

baseline_correct = form_guide["Correct"].iloc[train_end:test_end].to_numpy()
tree_correct = y_test.to_numpy() == predictions
accuracy_difference = tree_correct.mean() - baseline_correct.mean()
rng = np.random.default_rng(42)
bootstrap_differences = np.empty(10_000)
for iteration in range(len(bootstrap_differences)):
    sample = rng.integers(0, len(tree_correct), size=len(tree_correct))
    bootstrap_differences[iteration] = (
        tree_correct[sample].mean() - baseline_correct[sample].mean()
    )
lower, upper = np.percentile(bootstrap_differences, [2.5, 97.5])
print(f"Baseline accuracy (same test seasons): {baseline_correct.mean():.2%}")
print(f"Tree minus baseline accuracy: {accuracy_difference:.2%}")
print(f"Paired bootstrap 95% interval: [{lower:.2%}, {upper:.2%}]")

ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=tree.classes_)
plt.tight_layout()
plt.show()
