# Linear Regression Model 


from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
reg = LinearRegression()

model = LinearRegression(fit_intercept = False)


#cleaning data
data = pd.read_csv('house-prices_finalProject.csv') #read file 
tempX = data.drop(['RegionName','RegionID', 'SizeRank', 'Unnamed', 'Feb-20'], axis=1)
X = tempX.values
y = data['Feb-20']
mean_y = y.mean()
trained_model = model.fit(X, y)


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.45, shuffle=False)

# Training regression model
model = linear_model.LinearRegression()
trained_model = model.fit(X_train,y_train)
y_predicted = trained_model.predict(X_test)
print(y_predicted.mean())
print(mean_y)
plt.plot(X_test, y_predicted, color='black')
plt.show()


#KNN

from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
import matplotlib.pyplot as plt

# Cleaning data
data = pd.read_csv('house-prices_finalProject.csv') #read file 
tempX = data.drop(['RegionName','RegionID', 'SizeRank', 'Unnamed', 'Feb-20'], axis=1)
X = tempX.values
y = data['Feb-20']
mean_y = y.mean()
trained_model = model.fit(X, y)


# Loading data


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.45)
accuracy_list = []

for k in [1,3,5,7,9]:
    model = KNeighborsRegressor(n_neighbors=k, weights= 'uniform') # Define model
    model.fit(X_train, y_train) # Train model

    
    y_predicted = model.predict(X_test) #prediction
    MAE = mean_absolute_error(y_test, y_predicted)
    R2 = mean_squared_error(y_test, y_predicted)
    print("K value = ",k,"Mean Absolute Error = " ,MAE, "R2 = ", R2)
    
print(mean_y)
plt.plot(X_test, y_predicted, color='black')
plt.show()


#Decision Tree

import pandas as pd #import pandas package
from sklearn import tree #import sklearn tree package
import matplotlib.pyplot as plt #import matplot package
from sklearn.model_selection import train_test_split #import sklearn train test split package
from sklearn.tree import DecisionTreeClassifier #import slearn decision tree package 
from sklearn import datasets #import sklearn datasets package
from sklearn.metrics import accuracy_score #import sklearn accuracy score package
import numpy as np

#cleaning data
data = pd.read_csv('house-prices_finalProject.csv') #read file 
tempX = data.drop(['RegionName','RegionID', 'SizeRank', 'Unnamed', 'Feb-20'], axis=1)
X = tempX.values #formats without names
y = data['Feb-20']
mean_y = y.mean()
trained_model = model.fit(X, y)


feature_names = list(tempX.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45) #train test split size defined 

model = DecisionTreeClassifier(criterion = 'entropy') # Define model
model = model.fit(X_train, y_train) # Train model

y_predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)
print(y_predicted)
print(accuracy)
val = np.array(y).astype('str').tolist()
fig = plt.figure(figsize=(40,20)) #set figure size 
fig = tree.plot_tree(model, feature_names=feature_names,
class_names=val, filled=True)
plt.savefig('tree.pdf') #save figure 


#Neural Network

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cleaning data
data = pd.read_csv('house-prices_finalProject.csv') #read file 
tempX = data.drop(['RegionName','RegionID', 'SizeRank', 'Unnamed', 'Feb-20'], axis=1)
X = tempX.values
y = data['Feb-20']



# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.45)

# Defining model
tf.random.set_seed(1)
model = Sequential()

# Creating layers
input_layer = InputLayer(input_shape=(143,))
model.add(input_layer)
hidden_layer = Dense(9)
model.add(hidden_layer)
output_layer = Dense(1, activation='linear')
model.add(output_layer)

# Compiling
model.compile(loss = 'MeanAbsoluteError', metrics = 'Accuracy')

# Fitting model
model.fit(X_train, y_train, epochs = 10)
plt.plot(X_test, y_predicted, color='black')
plt.show()
