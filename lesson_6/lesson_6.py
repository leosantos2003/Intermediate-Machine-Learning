from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
train_path = Path(__file__).parent.parent / "train.csv"
test_path = Path(__file__).parent.parent / "test.csv"

X = pd.read_csv(train_path, index_col='Id')
X_test_full = pd.read_csv(test_path, index_col='Id')

# remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# "cardinality" means the number of unique values in a column
# select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# one-hot encode the data (pandas to shorten the code)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# --- BUILD MODEL ---

from xgboost import XGBRegressor

# define the model
my_model_1 = XGBRegressor(random_state=0)

# fit the model
my_model_1.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

# get predictions
predictions_1 = my_model_1.predict(X_valid)

mae_1 = mean_absolute_error(y_valid, predictions_1)

print(f"\n--- (1) Mean Absolute Error: {mae_1} ---\n")

# --- IMPROVE THE MODEL ---

# define the model
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# fit the model
my_model_2.fit(X_train, y_train)

# get predictions
predictions_2 = my_model_2.predict(X_valid)

# calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid)

print(f"\n--- (2) Mean Absolute Error: {mae_2} ---\n")

# --- BREAK THE MODEL ---

# define the model
my_model_3 = XGBRegressor(n_estimators=50, learning_rate=0.2, early_stopping_rounds=1)

# fit the model
my_model_3.fit(X_train, y_train,
               eval_set=[(X_valid, y_valid)],
               verbose=False)

# get predictions
predictions_3 = my_model_3.predict(X_valid)

# calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)

print(f"\n--- (3) Mean Absolute Error: {mae_3} ---\n")