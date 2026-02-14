from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# read the data
train_path = Path(__file__).parent.parent / "train.csv"
test_path = Path(__file__).parent.parent / "test.csv"

X = pd.read_csv(train_path, index_col='Id')
X_test = pd.read_csv(test_path, index_col='Id')

# remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print("\n--- (1) First five rows of the data: ---\n")
print(X_train.head())

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# --- FIRST METHOD: DROP COLUMNS WITH CATEGORICAL DATA ---

# drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("\n--- (2) MAE (drop categorical values): ---\n")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

print("\n--- (3) Unique values in 'Condition2' column in training data: ---\n")
print(X_train['Condition2'].unique())
print("\n--- (4) Unique values in 'Condition2' column in validation data: ---\n")
print(X_valid['Condition2'].unique())

# --- SECOND METHOD: ORDINAL ENCODING ---

# categorical columns in the training data
#object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
object_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

# columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if
                   set(X_valid[col]).issubset(set(X_train[col]))]

# problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

print("\n--- (5) Categorical columns that will be ordinal encoded: ---\n")
print(good_label_cols)
print("\n--- (6) Categorical columns that will be dropped from the dataset: ---\n")
print(bad_label_cols)

# drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print("\n--- (7) MAE from Approach 2 (Ordinal Encoding): ---\n") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

print("\n--- (8) Number of unique entries by column, in ascending order: ---\n")
print(sorted(d.items(), key=lambda x: x[1]))

# --- THIRD METHOD: ONE-HOT ENCODING ---

# How many categorical variables in the training data
# have cardinality greater than 10?
# high_cardinality_numcols = 3

# How many columns are needed to one-hot encode the 
# 'Neighborhood' variable in the training data?
# num_cols_neighborhood = 25

# columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print("\n--- (9) Categorical columns that will be one-hot encoded: ---\n")
print(low_cardinality_cols)
print("\n--- (10) Categorical columns that will be dropped from the dataset: ---\n")
print(high_cardinality_cols)

# apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# put thr removed index back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("\n--- (11) MAE from Approach 3 (One-Hot Encoding): ---\n") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))