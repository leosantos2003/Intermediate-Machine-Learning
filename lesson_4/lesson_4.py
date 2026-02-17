from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
train_path = Path(__file__).parent.parent / "train.csv"
test_path = Path(__file__).parent.parent / "test.csv"

X_full = pd.read_csv(train_path, index_col='Id')
X_test_full = pd.read_csv(test_path, index_col='Id')

# remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

# "cardinality" means the number of unique values in a column
# select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

print("\n--- (1) First five rows of the data: ---\n")
print(X_train.head())

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols),
                                               ('cat', categorical_transformer, categorical_cols)])

# define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# preprocessing of training data, fit model
clf.fit(X_train, y_train)

# preprocessing of training data, get predictions
preds = clf.predict(X_valid)

print(f"\n--- (2) MAE: {mean_absolute_error(y_valid, preds)} ---\n")

# --- IMPROVE THE PERFORMANCE ---

# preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median') # changed here from 'constant' to 'median'

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# define model
model = RandomForestRegressor(n_estimators=100, random_state=0) 

# bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# evaluate the model
score = mean_absolute_error(y_valid, preds)
print(f"\n--- (3) New MAE: {score} ---\n")

# --- GENERATE TEST PREDICTIONS ---

# preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# save test predictions to file
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('lesson_4/submission_lesson_4.csv', index=False)

print("\n--- (4) Test predictions saved to 'submission_lesson_4.csv' ---\n")