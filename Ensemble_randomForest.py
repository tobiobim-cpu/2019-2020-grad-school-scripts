##Esemble models are ML methods that combine several base models to produce one optimal predictive model
    #They combine decisions from multiple models to improve the overall performance
    #Involves creating a collection(or "ensemble") of multiple algorithms for the purpose of generating a single model
       #that's far more powerful and reliable than its component parts
    #It can be comprised of the same algorithm more than once (Random forest is an ensemble of decision trees)
        #Types of ensemble methods:
           #1)Max Voting: picks the results based on the majority votes from different models. Used in classification methods
           # 2)Averaging: running multiple models and then averaging the prediction. Used in both classification & regression
           # 3)Weighted averaging: multiple models to make predictions by allocating weights to different models predictions and averaging them out
           # 4)Bagging: results from multiple models and combines them to get a final result. Decision trees used frequently with bagging
             #Process overview: create subsets of the original data and run different models on the subsets; aggregate result; run the models in parallel.
           # 5)Boosting: takes results from multiple models and combines them to get a final result.
              #Process overview: SAME as bagging but it runs the models sequentially
                 #Six(6) steps are:
                    #1)Create a subset of the data.
                    #2)Run a model on the subset of the data and get predictions
                    #3)Calculate errors on these predictions
                    #4)Assign weights to the incorrect predictions
                    #5)Create another model with the same data and the next subset of data is created
                    #6)The cycle repeats until a strong learner is created.
    #Random Forest: is an ensemble model which follows the bagging method. Uses decision trees to form ensembles. This approach is useful for both classification and regression problems.
        #Five(5) steps are:
           #1)Create a random subset of the data.
           #2)Randomly select a set of features at each node in the decision tree
           #3)Decide the best split
           #4)For each subset of data, create a separate model(a "base learner")
           #5)Compute the final prediction by averaging the predictions from all the individual models.
        #Advanatages:
           #1)Easy to understand
           #2)Useful for data exploration
           #3)Reduced data cleaning(scaling not required)
           #4)Handles multiple data
           #5)Highly flexible and gives good accuracy
           #6)Works well on Large datasets
           #7)Overfitting is avoided
        #Disadvantages:
           #1)Not for continuous variables
           #2)Does not work well with sprse datasets
           #3)Computationally expensive
           #4)No interpretability

#%%
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)

y.columns = ['labels'] #you can also use 'target' for 'labels'
print(df.head(), '\n')
print(y[0:5])

#check to see if there are any missing values
df.isnull().any() == True

print(y.value_counts(), '\n') #Line 62 and 63 returns the same results
print(y.labels.value_counts())

#preparing the data for training the model
#split data
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=17)

####build the model
clsf = RandomForestClassifier(n_estimators=200, random_state=0)
##reformat the target data into an array so that it can work with the model
y_train_array = np.ravel(y_train) #the ravel() helps us to reformat the data into an horizontal array from a vertical array

clsf.fit(x_train, y_train_array)

y_pred = clsf.predict(x_test)

###Evaluate the model
print(metrics.classification_report(y_test, y_pred), '\n') #accuracy was 97%

#To visualize the 97% accuracy
y_test_array = np.ravel(y_test) #the ravel() helps us to reformat the data into an horizontal array from a vertical array
print((y_test_array), '\n')
print(y_pred) #only one value was wrong




# %%
