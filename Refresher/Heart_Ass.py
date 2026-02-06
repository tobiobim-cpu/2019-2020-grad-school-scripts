#Importing the necessary libraries
import pandas as pd
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image


#I am importing the dataset from my computer with the address below
heart = pd.read_excel(r'C:\Users\dolap\Exercise Files\Refresher\heart.xlsx')
print(heart.head())

features_column = ['sex', 'cp', 'fbs', 'restecg', 'exang']


# I need to differentiate my features from my labels
features = heart.drop('target', axis=1)  #This means everyother column except the 'target' column is a feature
labels = heart['target']   #This means 'target' is a label


#I need to split my dataset into training and test dataset
# x = features, y = labels. I am setting aside 80% for training set and 20% for test set.
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
print(len(labels), len(x_train), len(x_test))  #This is to check if the splitting was properly done

#Instantiate the learning model
#Fitting Classifier to the training set
clsf = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)

#Making predictions on the test data
y_pred = clsf.predict(x_test)

#Calculating the accuracy of the model
print("The accuracy of the model is:", str(round((accuracy_score(y_test, clsf.predict(x_test))), 3)) + '%\n')

print(confusion_matrix(y_test, y_pred))  #This prints the confusion matrix (This is the performance measurement for ML classification)
print(classification_report(y_test, y_pred))  #This prints the report of the classification

#For the confusion matrix:
   #Accuracy: [(TP + TN)/Total(N)]
   #Precision: [TP/((TP + FP) or predictive results)] -> Of all the PREDICTIVE POSITIVE classes, how much did we predict correctly
   #Recall/Sensitivity: [TP/((TP + FN) or actual results)] -> Of all the POSITIVE classes, how much did we predict correctly
   #Specificity: [TN / (TN + FP)] ->Of all the negative classes, how much did we predict correctly
   #F-Score: [(2*Recall*Precision) / (Recall + Precision)] -> It is difficult to compare models with different precision and recall
      #We use F-score to make them comparable. It's the HARMONIC MEAN of Precision and Recall. It should be HIGH!!!

#Visualizing the Decision Tree
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = StringIO()

export_graphviz(clsf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                feature_names=features_column, class_names=['0', '1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('heart.jpg')
Image(graph.create_jpg())

