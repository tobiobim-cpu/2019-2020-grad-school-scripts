##K_NN is a supervised classifier that memorizes observations from within a test set to predict classfication lablels for new, unlabeled observations
   #makes predictions based on how similar training observations are to the new, incoming observations
   #the more similar the observation values, the more likely they will be classified with the same label
   #Used cases:
      #1)stock price prediction 2)credit risk analysis 3)predictive trip planning 4)recommendation systems
    #Assumptions:
      #1)Dataset has little noise (2)Dataset is labeled (3)Dataset only contains relevant features
      #5)Dataset has distinguishable subgroups (6) Avoid using KNN on large datasets, it will prob take a long time

#%%
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

plt.style.use('seaborn-whitegrid')


cars = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

x_prime = cars[['mpg', 'disp', 'hp', 'wt']].values
y = cars.iloc[:,9].values

#Scale your variables before creting the model
x = preprocessing.scale(x_prime)

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

#Build the model
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

#Evaluate the model's prediction
y_pred = clf.predict(x_test)
y_expect = y_test

print(metrics.classification_report(y_expect, y_pred)) #Of all the points that were labeled 1, only 67% of the results that were returned were truly relevant
                                                       #of the entire dataset, 83% of the results that were returned were truly relevant.
#High precision + Low Recall -> Few results returned, but many of the label predictions that are returned as correct.
    #High accuracy but low completion                                                      

#For the confusion matrix:
   #Accuracy: [(TP + TN)/Total(N)]
   #Precision: [TP/((TP + FP) or predictive results)] -> Of all the PREDICTIVE POSITIVE classes, how much did we predict correctly
   #Recall/Sensitivity: [TP/((TP + FN) or actual results)] -> Of all the POSITIVE classes, how much did we predict correctly(measure of  model's completeness)
   #Specificity: [TN / (TN + FP)] ->Of all the negative classes, how much did we predict correctly
   #F-Score: [(2*Recall*Precision) / (Recall + Precision)] -> It is difficult to compare models with different precision and recall
      #We use F-score to make them comparable. It's the HARMONIC MEAN of Precision and Recall. It should be HIGH!!!





# %%
