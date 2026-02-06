##Naive Bayes is a ML method you can use to predict the likehood that an event will occur given evidence that's present in your data
   #Types of NB model:
      #1)Multinomial: When features are categorical or continuous
      #2)Bernoulli: For making prediction from binary features
      #3)Gaussian: Fro making prediction from Normally distributed features
   #Used cases: 1)Spam detection 2)Customer classification 3)Credit risk protection 4)Health risk protection
   #Assumptions:
      #1)Prredictors are independent of ech other
      #2)We will get wrong results if past conditions holds true but present circumstnces have changed

#%%
import pandas as pd
import numpy as np
import sklearn
import urllib

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

import urllib.request

raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=',')
print(dataset[0])

#isolate predictive variables
x = dataset[:,0:48]
#Labels records in the dataset as spam -> 1 and not spam -> 0
y = dataset[:,-1]
#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

#convert frequency count to variable
BernNB = BernoulliNB(binarize=True)
BernNB.fit(x_train, y_train)
print(BernNB, '\n')

y_expect = y_test
y_pred = BernNB.predict(x_test)

print("Bernoulli accuracy score is:", + accuracy_score(y_expect, y_pred), '\n')

#Multinommial
MultiNB = MultinomialNB()
MultiNB.fit(x_train, y_train)
print(MultiNB, '\n')

y_expect = y_test
y_pred = MultiNB.predict(x_test)

print("Multinomial accuracy score is:", + accuracy_score(y_expect, y_pred), '\n')

#Gaussian
GausNB = GaussianNB()
GausNB.fit(x_train, y_train)
print(GausNB, '\n')

y_expect = y_test
y_pred = GausNB.predict(x_test)

print("Gaussian accuracy score is:", + accuracy_score(y_expect, y_pred), '\n')

#To improve the score for BernNB, if we set our binarize parameter to 0.1, we get optimal results
BernNB = BernoulliNB(binarize=0.1)
BernNB.fit(x_train, y_train)
print(BernNB, '\n')

y_expect = y_test
y_pred = BernNB.predict(x_test)

print("Improved_Bernoulli accuracy score is:", + accuracy_score(y_expect, y_pred), '\n')


# %%
