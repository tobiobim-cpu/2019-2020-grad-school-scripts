import pandas as pd
import numpy as np
import sklearn
import scipy
import urllib
import matplotlib.pyplot as plt

from pylab import rcParams
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.style.use('seaborn-whitegrid')


cars = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

x_prime = cars[['mpg', 'disp', 'hp', 'wt']].values
y = cars.iloc[:,9].values
print('y is', y)
print(cars.head())

#Scale your variables/features before creating the model
x = preprocessing.scale(x_prime)
print(x)

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

#build the pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('knn', KNeighborsClassifier()),
])

pipe.fit(x_train, y_train)

#Evaluate the model's prediction
y_pred = pipe.predict(x_test)
y_expect = y_test

print(metrics.classification_report(y_expect, y_pred))