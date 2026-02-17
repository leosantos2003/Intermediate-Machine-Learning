from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# read the data
train_path = Path(__file__).parent.parent / "train.csv"
test_path = Path(__file__).parent.parent / "test.csv"

X_full = pd.read_csv(train_path, index_col='Id')
X_test_full = pd.read_csv(test_path, index_col='Id')

# remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

#  use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print("\n--- (1) First five rows of the data: ---\n")
print(X_train.head())

print("\n--- (2) Shape of training data (num_rows, num_columns): ---\n")
print(X_train.shape)

missing_val_count_by_column = (X_train.isnull().sum())
print("\n--- (3) Number of missing values in each column of training data: ---\n")
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# number of rows in the training data
num_rows = X_train.shape[0]

# number of columns with missing values
num_cols_with_missing = X_train.isnull().any().sum()

# number of missing entries contained in all of the training data
tot_missing = X_train.isnull().sum().sum()

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# --- FIRST METHOD: DROP COLUMNS WITH MISSING VALUES ---

# names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("\n--- (4) MAE (drop columns with missing values): ---\n")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# --- SECOND METHOD: IMPUTATION ---

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# putting the removed columns names by the imputation back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("\n--- (5) MAE (imputation): ---\n")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# --- THIRD METHOD: OF MY CHOOSING ---

# preprocessed training and validation features
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# get validation prediction and MAE
preds_valid = model.predict(final_X_valid)
print("\n--- (6) MAE (my approach): ---\n")
print(mean_absolute_error(y_valid, preds_valid))

# preprocess test data
final_X_test = pd.DataFrame(final_imputer.fit_transform(X_test))

# get test predictions
preds_test = model.predict(final_X_test)

# save test predictions to file
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('lesson_2/submission_lesson_2.csv', index=False)