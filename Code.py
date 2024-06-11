import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

income_df = pd.read_csv("adult.csv")
income_df.replace({"?": np.nan}, inplace=True)
for threshold, col_name in zip([1500, 5000, 200], ["workclass", "occupation", "native-country"]):
    counts = income_df[col_name].value_counts()
    repl = counts[counts <= threshold].index
    income_df[col_name].replace(repl, "other {}".format(col_name), inplace=True)
income_df["income"].replace({"<=50K": 0, ">50K": 1}, inplace=True)
income_df["gender"].replace({"Male": 1, "Female": 0}, inplace=True)
income_df.drop(columns=["fnlwgt", "education"], inplace=True)
print(repl)
###### TRAIN TEST SPLIT ######

X = np.array(income_df.copy().drop(columns=["income"]))
y = np.array(income_df["income"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

###### COLUMN TRANSFORMER #######
OHE = OneHotEncoder(sparse=False, drop="first", dtype='int64')
trf_1 = ColumnTransformer(remainder='passthrough', transformers=[
    ('One Hot Encoder', OHE, [1, 3, 4, 5, 6, 11])
])

###### RANDOM FOREST CLASSIFIER #######
clf = RandomForestClassifier(random_state=42)
param_dict = {"n_estimators": [5, 7, 9, 11], "max_depth": [3, 5, 7]}
model = GridSearchCV(clf, param_grid=param_dict, cv=5)

###### PIPELINE ######
pipeline_1 = Pipeline([
    ("Column Transformer", trf_1),
    ("Classifier", model)
])

###### EVALUATION ON TRAIN DATA ######
pipeline_1.fit(X_train, y_train)
y_pred = pipeline_1.predict(X_train)
error = np.sqrt(mean_squared_error(y_train, y_pred))

###### EVALUATION ON TEST DATA ######
y_pred_2 = pipeline_1.predict(X_test)
error_2 = np.sqrt(mean_squared_error(y_test, y_pred_2))

###### PRINT STATEMENTS ######
print(income_df.info())
print("################################################")
#print(income_df["marital-status"].value_counts())
#print(income_df["relationship"].value_counts())
#print(income_df["race"].value_counts())
#print(income_df["gender"].value_counts())
print("Best Classifier: ", pipeline_1["Classifier"].best_estimator_)
print("Best parameters of Random Forest: ", pipeline_1["Classifier"].best_params_)
print("Best score of Random Forest: ", pipeline_1["Classifier"].best_score_)
print("Error2: ", error_2)

###### DATA VISUALIZATION ######
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
ax1.hist(income_df["age"], bins=50, alpha=0.2)
ax1.set_xlabel("age")
ax2.hist(income_df["capital-gain"], bins=50, alpha=0.2)
ax2.set_xlabel("capital-gain")
ax3.hist(income_df["capital-loss"], bins=50, alpha=0.2)
ax3.set_xlabel("capital-loss")
ax4.hist(income_df["hours-per-week"], bins=50, alpha=0.2)
ax4.set_xlabel("hours-per-week")
#fig.subplots_adjust(wspace=0, hspace=0)
plt.show()
#plt.close()
