from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
train_path = Path(__file__).parent.parent / "train.csv"
test_path = Path(__file__).parent.parent / "test.csv"

train_data = pd.read_csv(train_path, index_col='Id')
test_data = pd.read_csv(test_path, index_col='Id')

# remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# select numeric columns only
numeric_cols = [cname for cname in train_data.columns
                if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

print("\n--- (1) First five rows of the data: ---\n")
print(X.head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50, random_state=0))])

from sklearn.model_selection import cross_val_score

# multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print(f"\n--- (2) Avarage MAE score: {scores.mean()} ---\n")

# --- WRITING A USEFUL FUNCTION ---

def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators, random_state=0))])

    scores = -1 * cross_val_score(my_pipeline, X, y,
                                 cv=3,
                                 scoring='neg_mean_absolute_error')

    return scores.mean()

# --- TESTING DIFFERENT PARAMETER VALUES ---

results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))

plt.title("Model Performance vs. Estimators Number")
plt.xlabel("n_estimators")
plt.ylabel("Score (MAE)")

import os

lesson_5_path1 = os.path.join("lesson_5", "model_performance_vs_estimators_number.png")
plt.savefig(lesson_5_path1)
plt.close()
print("\n--- 'model_performance_vs_estimators_number.png' generated! ---\n")

# --- FINDING THE BEST PARAMETER VALUE ---

n_estimators_best = 200