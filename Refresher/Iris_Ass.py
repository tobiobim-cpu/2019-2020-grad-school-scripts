from math import*
import pandas as pd  #Importing the pandas library
from sklearn.model_selection import train_test_split  #importing the train_test_split module from the Scikit library
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier  #Importing the KNN classifier from Scikit library
from sklearn.metrics import accuracy_score  #Importing the accuracy module from from Scikit library
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


#I am importing the dataset from my computer with the address below
iris = pd.read_excel(r'C:\Users\dolap\Exercise Files\Refresher\iris.xlsx')
print(iris.head())
#features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

#Since we have categorical labels, KNN doesnt accept string labels. So we need to transform them into numbers
#It will return 0 for Setosa, 1 for Versicolor and 2 for Virginica
variety_num = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
iris['variety'] = iris['variety'].map(variety_num)

#I need to split my dataset into training and test dataset
features = iris.drop('variety', axis=1)  #This means everyother column except the 'variety' column is a feature
labels = iris['variety']  #This means 'variety' is a label or target

# x = features, y = labels. I am setting aside 80% for training set and 20% for test set.
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(len(labels), len(x_train), len(x_test))  #This is to check if the splitting was properly done

#Fitting Classifier to the training set
#Instantiate the learning model at k =7
clsf = KNeighborsClassifier(n_neighbors=7).fit(x_train, y_train)


#The distance criterion choosen was euclidean distance
def euclidean_distance():

    return sqrt(sum(pow(a - b, 2) for a, b in zip(clsf.predict(x_test), y_test)))

print(euclidean_distance())  #Prints out the value of the euclidean distance

#Making predictions on the test data
y_pred = clsf.predict(x_test)

#Calculating the accuracy of the model
print("The accuracy of the model is:", str(round((accuracy_score(y_test, clsf.predict(x_test))), 3)) + '%\n')

print(confusion_matrix(y_test, y_pred))  #This prints the confusion matrix (Performance measurement for ML classifications)
print(classification_report(y_test, y_pred), '\n')  #This prints the report of the classification

#To visualize our accuracy score of 93%
print(y_pred, '\n')
import numpy as np
y_expect = np.ravel(y_test)
print(y_expect)
print(len(y_expect))

scores = cross_val_score(clsf, features, labels, cv=10)
print(scores)
print(scores.mean())

#1. Plot Accuracy vs. k (Hyperparameter Tuning)
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, features, labels, cv=10)
    cv_scores.append(scores.mean())

plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Accuracy vs. k')
plt.grid(True)
plt.show()








