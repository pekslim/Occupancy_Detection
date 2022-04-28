from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dtest1 = pd.read_csv('Data/datatest.txt', header=0)
dtrain = pd.read_csv('Data/datatraining.txt', header=0)
dtest2 = pd.read_csv('Data/datatest2.txt', header=0)

dtest1.insert(len(dtest1.columns) - 1, 'CO2diff', dtest1['CO2'].diff(2))
dtest1 = dtest1.dropna()
nr1 = len(dtest1)
dtrain.insert(len(dtrain.columns) - 1, 'CO2diff', dtrain['CO2'].diff(2))
dtrain = dtrain.dropna()
nr2 = len(dtrain)
dtest2.insert(len(dtest2.columns) - 1, 'CO2diff', dtest2['CO2'].diff(2))
dtest2 = dtest2.dropna()
nr3 = len(dtest2)

df = pd.concat([dtest1, dtrain, dtest2])
df.drop('no', axis=1, inplace=True)
df = df.reset_index(drop=True)
n_feature = df.values.shape[1]

# df.to_csv('Data/FullData.csv')
df["date"] = pd.to_datetime(df["date"])
# df['NS'] = df["date"].dt.hour * 60 * 60 + df["date"].dt.minute * 60 + df["date"].dt.second
timecol = df["date"].dt.round("H")
df["hour"] = timecol.dt.time
df["date"] = timecol.dt.date
df["weekday"] = pd.DatetimeIndex(df['date']).weekday
df['WeekStatus'] = np.where((df['weekday'] >= 5), 0, 1)
df.drop('weekday', axis=1, inplace=True)

# print(df)

df = pd.get_dummies(df, columns=['hour'])
# df = pd.get_dummies(df, columns=['weekstatus'])
# df = pd.get_dummies(df, columns=['weekday'])
time_indices = df.columns[n_feature:]

########################### Data Partition ##############################

# tr_data = df.iloc[: int(0.6 * len(df)), :]
# # print(tr_data['Occupancy'].value_counts(normalize=True))
# ts_data1 = df.iloc[int(0.6 * len(df)):, :]
# # print(ts_data['Occupancy'].value_counts(normalize=True))

# ts_data1 = df.iloc[: int(0.4 * len(df)), :]
# # print(ts_data['Occupancy'].value_counts(normalize=True))
# tr_data = df.iloc[int(0.4 * len(df)):, :]
# # print(tr_data['Occupancy'].value_counts(normalize=True))


ts_data1 = df.iloc[:nr1, :]
# print(ts_data1)
# print(dtest1['Occupancy'].value_counts(normalize=True))
tr_data = df.iloc[nr1:nr1+nr2, :]
# print(tr_data)
# print(dtrain['Occupancy'].value_counts(normalize=True))
ts_data2 = df.iloc[nr1+nr2:len(df), :]
# print(ts_data2)
# print(dtest2['Occupancy'].value_counts(normalize=True))

X_env = tr_data[["CO2", "Temperature"]]
X_train = pd.concat([X_env,tr_data[time_indices]], axis=1)
y_train = tr_data["Occupancy"]

X_env = ts_data1[["CO2", "Temperature"]]
X_test1 = pd.concat([X_env,ts_data1[time_indices]], axis=1)
y_test1 = ts_data1["Occupancy"]

X_env = ts_data2[["CO2", "Temperature"]]
X_test2 = pd.concat([X_env, ts_data2[time_indices]], axis=1)
y_test2 = ts_data2["Occupancy"]

feature_name = [ "CO2", "Temperature"]
feature_name += time_indices.values.tolist()

#################################### Training ##################################

# # Decision Tree
# mod = DecisionTreeClassifier(criterion='entropy', max_depth=10)
# mod.fit(X_train.values, y_train.values)
# # fig = plt.figure(figsize=(250,200))
# # _ = plot_tree(mod, feature_names=feature_name, filled=True)
# # fig.savefig("decistion_tree.png")


# Random Forest
mod = RandomForestClassifier()
mod.fit(X_train, y_train)
plt.figure(figsize=(10,6))
plt.barh(feature_name, mod.feature_importances_)
print(sum(mod.feature_importances_[2]))
plt.title("RF Feature Importance",fontsize=18)
plt.show()
# fig = plt.figure(figsize=(250,200))
# _ = plot_tree(mod.estimators_[0], feature_names=feature_name, filled=True)
# fig.savefig("decistion_tree.png")


# # Random Forest
# mod = GradientBoostingClassifier()
# mod.fit(X_train, y_train)


# # SVM
# mod = SVC()
# mod.fit(X_train, y_train)


#################################### Testing ##################################

pred_train_rf = mod.predict(X_train)
print('Training: ', accuracy_score(y_train, pred_train_rf))
pred_test_rf = mod.predict(X_test1)
print('Testing1: ', accuracy_score(y_test1, pred_test_rf))
pred_test_rf = mod.predict(X_test2)
print('Testing2: ', accuracy_score(y_test2, pred_test_rf))





