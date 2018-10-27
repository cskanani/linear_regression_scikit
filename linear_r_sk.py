import numpy as np
np.warnings.filterwarnings('ignore') #for supressing warnings from numpy
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

data = np.loadtxt('winequality-red.csv',skiprows=1,delimiter=';') #reading data file into a np array

#saperating features from input data
x = np.concatenate((data[:, 0:10],data[:,11].reshape(-1,1)), axis=1)

#adding square root of each feature as a new feature as it is improving accuracy of model
funt = FunctionTransformer(np.sqrt)
x = np.concatenate((x,funt.fit_transform(x)), axis=1)

#separating result variable from input data
y = data[:,10]


#splitting data into train and test samples and training linear regression model
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.20)
linear_r = LinearRegression(normalize=True)
linear_r.fit(x_train,y_train)
cross_val_mean = cross_val_score(linear_r, x_train, y_train, scoring='neg_mean_squared_error',cv=5)

print('Predicted AC   vs   Actual AC')
for x,y in zip(linear_r.predict(x_test),y_test):
    print('{:8.3f} {:18.3f}'.format(x,y))

    
print('\nRSS for each fold : ')
print(abs(cross_val_mean)*x_train.shape[0]) #multiplying mean squared error with number of samples used for predicting it to get residual sum of squares

print('\nAverage RSS : ')
print(sum(abs(cross_val_mean)*x_train.shape[0])/5)
