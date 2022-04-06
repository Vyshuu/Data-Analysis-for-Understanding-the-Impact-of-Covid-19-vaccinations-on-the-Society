import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt

dataset = pd.read_csv('Dataset/vaccinations.csv')
print(dataset.head())
dataset['date'] = pd.to_datetime(dataset['date']).dt.strftime('%Y-%m-%d') 
dataFrame = pd.pivot_table(data=dataset, values='total_vaccinations', index='vaccine', columns='date', aggfunc='sum', fill_value=0)


year_wise_sale = dataset.groupby(['vaccine'])['total_vaccinations'].sum()
year_wise_sale.plot(figsize=(15, 6)) 
plt.title("Vaccines Manufacturing")
plt.xlabel('Manufacturers')
plt.ylabel('Vaccines')
plt.show()


sns.factorplot(data = dataset, x ="location", y = "total_vaccinations", row = "vaccine")
plt.title("Location Wise Manufacturing")
plt.show()

x_train = 8
y_train = 1
y_test = 8

dataset = dataFrame.values
time_periods = dataset.shape[1]
lag_loops = time_periods + 1 - x_train - y_train - y_test

training = []
for i in range(lag_loops):
    value = dataset[:,i:i+x_train+y_train]
    training.append(value)
training = np.vstack(training)
Xtrain, Ytrain = np.split(training,[x_train],axis=1)
 
max_column_test = time_periods - x_train - y_train + 1
testing = []
for i in range(lag_loops,max_column_test):
    testing.append(dataset[:,i:i+x_train+y_train])
testing = np.vstack(testing)
Xtest, Ytest = np.split(testing,[x_train],axis=1)
 
if y_train == 1:
    Ytrain = Ytrain.ravel()
    Ytest = Ytest.ravel()

tree = DecisionTreeRegressor() 
tree.fit(Xtrain,Ytrain)

prediction = tree.predict(Xtest) 

actual = []
forecast = []
i = len(Ytest)-1
index = 0
while i > 0:
    actual.append(Ytest[i])
    forecast.append(prediction[i])
    print('Day=%d, Forecasted=%f, Actual=%f' % (index+1, prediction[i], Ytest[i]))
    index = index + 1
    i = i - 1
    if len(actual) > 30:
        break

rmse = sqrt(mean_squared_error(Ytest,prediction))
print('\n\nRMSE : ',round(rmse,1))

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Actual & Forecast Vaccines Manufacturing')
plt.ylabel('Manufacturing Count')
plt.plot(actual, 'ro-', color = 'blue')
plt.plot(forecast, 'ro-', color = 'green')
plt.legend(['Required Vaccines', 'Forecasted Vaccines'], loc='upper left')
#plt.xticks(wordloss.index)
plt.title('Covid19 Vaccines Forecasting Graph')
plt.show()


