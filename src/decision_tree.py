"""Train and evaluate a decision tree on prepared match form features."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "processed" / "prem_form_guide.csv"

form_guide = pd.read_csv(DATA_FILE)
feature_columns = [
    "Home Form",
    "Away Form",
    "Home Attack Form",
    "Home Defense Form",
    "Away Attack Form",
    "Away Defense Form",
]
X = form_guide[feature_columns]
y = form_guide["Result"]

train_end = form_guide[form_guide["Season"] == "2016-2017"].index[0]
test_end = form_guide[form_guide["Season"] == "2018-2019"].index[0]
X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]

tree = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

print(f"Training accuracy: {tree.score(X_train, y_train):.2%}")
print(f"Test accuracy (2016–2018): {tree.score(X_test, y_test):.2%}")

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_columns, class_names=tree.classes_, filled=True)
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=tree.classes_)
plt.tight_layout()
plt.show()
