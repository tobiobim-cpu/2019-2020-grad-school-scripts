##Perceptron is a NN with just one layer. Its a linear classifier that outputs a binary response variable
  #Consequently, the algorithm is called a linear binary classifier.
    #Data is said to have "linear separability" if it can be cleanly classified into one of two classes
       #Your data must be linearly separable in order for a perceptron to operte properly
##4-elements of a perceptron:
   #1)Input layer 20 Weights and bias 3)Weighted sum 4) Activation function
      #Activation function:
        #1) is a mathematical func that is deployed on each unit in a NN
        #2) all units in a shared layer deploy the same activation function
        #3) purpose of activation functions is to enable NN to model complex, nonlinear phenomenon
        #Types of activation func:
           #1)Linear activation: tf.matmul() -> single layer perceptron
           #2)Logistic sigmoid: used often in the final output layer, useful with binary inpt features
           #3)Threshold function: useful with binary features
           #4)ReLU (rectified linear unit)
           #5)SoftMax


#%%
import pandas as pd
import numpy as np
import sklearn

import matplotlib.pyplot as plt 
from pandas import Series, DataFrame
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()
x = iris.data
y = iris.target

#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
##Normalize the data
standardize = StandardScaler()
standardized_x_test = standardize.fit_transform(x_test)
standardized_x_train = standardize.fit_transform(x_train)

##Create the perceptron such that it passes through the data 50 times & learning rate =0.15
perceptron = Perceptron(max_iter=50, eta0=0.15, tol=1e-3, random_state=15)

#Train the model
perceptron.fit(standardized_x_train, y_train.ravel()) #Using the ravel() so that it would format it properly for the model

#Lets make a prediction based on our train model on the test data
y_pred = perceptron.predict(standardized_x_test)
#print(y_test)
plt.show()
print(y_pred)
plt.show()

print('Classification report:', classification_report(y_test, y_pred))



#For the confusion matrix:
   #Accuracy: [(TP + TN)/Total(N)]
   #Precision: [TP/((TP + FP) or predictive results)] -> Of all the PREDICTIVE POSITIVE classes, how much did we predict correctly
   #Recall/Sensitivity: [TP/((TP + FN) or actual results)] -> Of all the POSITIVE classes, how much did we predict correctly
   #Specificity: [TN / (TN + FP)] ->Of all the negative classes, how much did we predict correctly
   #F-Score: [(2*Recall*Precision) / (Recall + Precision)] -> It is difficult to compare models with different precision and recall
      #We use F-score to make them comparable. It's the HARMONIC MEAN of Precision and Recall. It should be HIGH!!!





# %%
