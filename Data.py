import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score

data1 = pd.read_csv('Data/datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = pd.read_csv('Data/datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = pd.read_csv('Data/datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)

data1.insert(len(data1.columns)-1, 'CO2diff', data1['CO2'].diff(2))
data2.insert(len(data2.columns)-1, 'CO2diff', data2['CO2'].diff(2))
data3.insert(len(data3.columns)-1, 'CO2diff', data3['CO2'].diff(2))

n_features = data1.values.shape[1]
print(n_features)
plt.figure()
for i in range(1,n_features):
    plt.subplot(n_features,1,i)
    if data1.columns.values[i] == "CO2diff":
        plt.ylim((-50, 50))
    plt.plot(data1.index, data1.values[:, i])
    plt.plot(data2.index, data2.values[:, i])
    plt.plot(data3.index, data3.values[:, i])
    plt.title(data1.columns[i], y=0.5, loc='right')
plt.show()

####################################### Dataset Output #########################################

# data1.drop('no', axis=1, inplace=True)
# data1.to_csv('Data/datatest.csv')
# data2.drop('no', axis=1, inplace=True)
# data2.to_csv('Data/datatraining.csv')
# data3.drop('no', axis=1, inplace=True)
# data3.to_csv('Data/datatest2.csv')

# df = pd.concat([data1.dropna(), data2.dropna(), data3.dropna()])
# df.drop('no', axis=1, inplace=True)
# df = df.reset_index(drop=True)
# df.to_csv('Data/FullData.csv')

data = pd.concat([data1, data2, data3])
data.drop('no', axis=1, inplace=True)
print(data)
# data.to_csv('Data/AllData.csv')

# data_test = pd.concat([data1, data3])
# data_test.drop('no', axis=1, inplace=True)
# data_test.to_csv('Data/TestData.csv')

# data_train = pd.concat([data2])
# data_train.drop('no', axis=1, inplace=True)
# data_train.to_csv('Data/TrainData.csv')

##################################### Correlation Matrix #######################################

plt.figure(figsize=(10,6))
# df = pd.DataFrame(data2, columns=["Temperature","Humidity","CO2","HumidityRatio","Occupancy"])
df = pd.DataFrame(data, columns=["Temperature","Humidity","CO2","HumidityRatio","Occupancy"])
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


##################################### ZeroRule Accuracy #######################################
data_test = pd.concat([data3])
tvalues = data_test.values
X, y = tvalues[:, :-1], tvalues[:, -1]

def naive_prediction(testX, value):
    return [value for x in range(len(testX))]

# evaluate skill of predicting each class value
for value in [0, 1]:
    yhat = naive_prediction(X, value)
    score = accuracy_score(y, yhat)
    print('Naive=%d score=%.3f' % (value, score))
