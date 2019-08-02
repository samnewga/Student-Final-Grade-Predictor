#pandas is data structures for data analysis, time series,and statistics
#numpy works for array processing for numbers, strings, records, and objects.
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

print("Student Final Grade Prediction \n"
      "Project description: A program that takes hand picked variables to predict their final grade result \n"
      "Variable list: 1st test grade, 2nd test grade, studytime, failures, absences, health, weekday and weekend alcohol consumption \n")
#this is the data set we are using and how we seperate each variable
data = pd.read_csv("student-mat.csv", sep=";")

#this is the variables we are using from our data set
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "health" , "Dalc", "Walc"]]

#what we plan to predict
predict = "G3"

#x and y values
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#this allows us to not use the whole data set which would let the code see the answers by breaking it down
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#training model being used
linear = linear_model.LinearRegression()

#fits x and y train data to find the best fit line
linear.fit(x_train, y_train)

#scores x and y test
linear.score(x_test, y_test)

#returns the accuracy of the model
accuracy = linear.score(x_test, y_test)

#prints out the accuracy
print('Accuracy of prediction: \n', accuracy)

#prints list of coefficients
print('\nCoefficients: \n', linear.coef_)

#prints the intercept
print('\nIntercept: \n', linear.intercept_, '\n')

#going to take arrays and do predictions and guess on the the test data
predictions = linear.predict(x_test)

#prediction model
for x in range(len(predictions)):
    #prints out the prediction, input data and the actual result of the test from the data set "G3"
    print('Prediction:', predictions[x],'- Real final grade result:', y_test[x], '- Variables used: ', x_test[x])

