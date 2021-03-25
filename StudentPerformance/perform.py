import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# read the csv file
data = pd.read_csv("student-mat.csv", sep=';')

# select the required attributes or the ones you want
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


predict = 'G3'

# label is the one you are trying to predict and features are the ones you are gonna be predicting from

x = np.array(data.drop([predict], 1))  # features
y = np.array(data[predict])  # labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# splitting the data into training and testing data
best = 0
"""
for _ in range(40):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # making the linear regression model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    print(acc)
    if(acc>best):
        with open("student.pickle", "wb") as f:
            pickle.dump(linear, f)
"""
pickle_in = open("student.pickle", "rb")
linear = pickle.load(pickle_in)

# coef are the m values in the multi dem line
print('co :', linear.coef_)
# intercept is the y intercept
print(' intercept : ', linear.intercept_ )

#
# prediction = linear.predict(x_test)
# for x in range(len(prediction)):
#     print(prediction[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()
