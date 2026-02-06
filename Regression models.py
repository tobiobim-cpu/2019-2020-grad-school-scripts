##Linear regression is a statistical ML method you can use to quantify, and make predictions based on, relationships between numerical variables.
##It can be used for 1) Sales forecasting 2)Supply cost forecasting, 3) Resource consumption Forecasting 4) Telecom Services Lifecycle Forecasting
##Assumptions:
      #All variables are continous numeric, not categorical.
      #Data is free of missing values and Outliers.
      #There's a linear relationship btw predictors(x) and predictant (y).
      #All predictors are independent of each other.
      #Residuals(prediction errors) are normally distributed.

#%%
#SIMPLE LINEAR REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sb
sb.set_style('whitegrid') #Setting the style for seaborn

from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter ##

rcParams['figure.figsize'] = 5, 4#dimensions for data visulaization

rooms = 2*np.random.rand(100,1)+3
#print(str(rooms[1:10]) + '\n') #prints the first 9 numbers

price = 265 + 6*rooms + abs(np.random.randn(100,1))
#print(price[1:10])

##To visualize in scatter plot
plt.plot(rooms, price, 'r*')
plt.xlabel('Numbers of rooms, 2019 Average')
plt.ylabel('2019 Average Home price (1000s USD')
plt.show()

x = rooms
y = price

LinReg = LinearRegression() #assigning a variable "LinReg" to the Linear Regression function
LinReg.fit(x,y) #Fit the model to the data
print(LinReg.intercept_, LinReg.coef_) #intercept = 266 & Coefficient = 5.9
## Regression equation -> y = mx + b -> y = 5.9x + 266

#To see how well the model performs
print(LinReg.score(x,y)) #0.96

###MULTIPLE LINEAR REGRESSION
enroll = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\enrollment_forecast.csv')
enroll.columns = ['year', 'roll', 'unem', 'hgrad', 'inc']
print(enroll.head()) #Data from New Mexico -> Under year -> 1 = 1961, roll = enrollment numbers, unem = unemployment rate, hgrad = graduation rate, inc = income 

#to check for correlation btw variables
sb.pairplot(enroll) #Visualize to see if there's any linear relationship of some sort
plt.show()
print(enroll.corr()) #hgrad and unem are not showing linear correlation
#Lets see how they do as predictor with the linear regression ,model
enroll_data = enroll[['unem', 'hgrad']].values #we want to extract the values of these variables

enroll_target = enroll[['roll']].values

enroll_data_names = ['unem', 'hgrad']
#Before using our variables as predictors in a linear regression model, we should SCALE THEM!
x, y = scale(enroll_data), enroll_target
#We need to check for missing values
missing= []
X = x==np.NAN
if (missing == X):
    x_missing = x[X == True]
    print(x_missing)
else:
    print("No missing values")

"""""
missing_values = x==np.NAN
x_missing = x[missing_values == True]
print(x_missing) #print out the missing values in x
"""""
#Instantiate the Linear regression
LineReg = LinearRegression(normalize=True) #It tells linear regression to normalize our variables before regression

LineReg.fit(x, y) #fit the model to the data
print(str(LineReg.score(x,y)),  '\n') #To see how well the model performs -> 0.848 (R-square: It measures how well the regression line fits the data)

##LOGISTIC REGRESSION
   #It's a simple ML method you can use to predict the value of numeric categorical variable based on its relationship with predictor variables
   #You're predicting categories for ordinal variables (Linear -> predicting values for numeric continuous variables)
   #Use Cases: 1)Customer churn Prediction 2)Employee Attrition Modeling 3)Hazardous Event Prediction 4)Purchase propensity vs Spend analysis
   #Assumptions: 1) Data is free of missing values.
                #2) The predictant variable is binary(only accepts 2 values) or Ordinal (categorical variable with ordered values)
                #3) All predictors are independent of each other
                #4) There are at least 50 observations per predictor variable (to ensure reliable results)

#Logistic Regression on Titanic dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix #(This is the performance measurement for ML classification)
from sklearn.metrics import precision_score, recall_score, accuracy_score

##Variable descriptions:
    #Survived -> Survival(0=No, 1=Yes)
    #Pclass -> Passenger Class(1=1st, 2=2nd, 3=3rd)
    #SibSp -> Number of Siblings/Spouses Aboard
    #Parch -> Number of Parents/Children Aboard
    #Fare -> Passenger Fare(British pound)
    #Embarked -> Where they boarded the ship:
        #C -> Cherboug, France
        #Q -> Queenstown, UK
        #S -> Southampton-Cobh, Ireland

titanic_training = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\titanic-training-data.csv')

titanic_training.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',  
                                             'Fare', 'Cabin', 'Embarked']
print(titanic_training.head(), '\n')
print(titanic_training.info()) #To get an overall info about the data & to see if there are missing values

#To check if your target variable is binary
    #We want to predict "Survival(0=No, 1=Yes" -> Target)
sb.countplot(x='Survived', data=titanic_training, palette='hls') #'hls' -> Hue(colour), saturation, and lightness
plt.show()
#To check for missing values
titanic_training.isnull().sum()
#To check the count of rows
titanic_training.describe()
######Handling missing values and dropping irrelevant variable
titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis=1)
print(titanic_data.head())

#Get a quick overview of the distribution of the datapoint of 'parch nd age'
sb.boxplot(x='Parch', y='Age', data=titanic_data, palette='hls')
plt.show()
#Find the average age/parch category
Parch_groups = titanic_data.groupby(titanic_data['Parch'])
Parch_groups.mean() #It means the average age of those without parent or children is 32

#Imputing the missing values
def age_approx(cols):
    Age = cols[0]
    Parch = cols[1]

    if pd.isnull(Age): #Loop through the DF under 'AGE' for each of the 'Parch' categories (0-6), 
                             #if 'AGE column' is empty, insert the average age for each 'Parch categpory'
        if Parch == 0:
            return 32
        elif Parch == 1:
            return 24
        elif Parch == 2:
            return 17
        elif Parch == 3:
            return 33
        elif Parch == 4:
            return 45
        else:
            return 30 #overall AGE mean of passengers on the ship 
                      #(I guess it's bcoz the last two Parch category were less than 45)
    else:
        return Age
        
titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
titanic_data.isnull().sum()

#Since we still have 2 missing values for 'Embarked', we can just drop those 2. It wont impact much since we have 891 datapoints
titanic_data.dropna(inplace=True)
#we have to Reset our index after dropping the values so that we have an accurate index for our output dataset
titanic_data.reset_index(inplace=True, drop=True)
print(titanic_data.info()) #We now have 889 datapoints because we dropped those '2' datapoints

#####Re-encoding variable
#Converting categorical variables to a dummy indicator so that they can work within the model
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
##Convert gender variable to numeric
gender_cat = titanic_data['Sex']
gender_encoded = label_encoder.fit_transform(gender_cat) ##Similar to line 15&16 in iris.py
#1=Male, 0=Female
gender_DF = pd.DataFrame(gender_encoded, columns=['male_gender'])
#gender_DF.head()

##Convert embarked variable to numeric
embarked_cat = titanic_data['Embarked']
embarked_encoded = label_encoder.fit_transform(embarked_cat)
#0=C, 1=Q, 2=S (Not Yet Binary but Multi-nomial categorical variable-> See line 111
##But We need 'Embarked' to be Binary, so we need to use 'onehot encoder'
from sklearn.preprocessing import OneHotEncoder
binary_encoder = OneHotEncoder(categories='auto')
embarked_1hot = binary_encoder.fit_transform(embarked_encoded.reshape(-1,1))
embarked_1hot_mat = embarked_1hot.toarray()
embarked_DF = pd.DataFrame(embarked_1hot_mat, columns=['C', 'Q', 'S'])
embarked_DF.head() #It is now Binary

#Dropping 'Sex' because we now have 'gender_DF' 
#Dropping 'Embarked' bcos we now have 'embarked_DF'
titanic_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

#Now we have concatenate the new encoded variables
titanic_dmy = pd.concat([titanic_data, gender_DF, embarked_DF], axis=1, verify_integrity=True).astype(float)
titanic_dmy.head()

###Validating data
##Check for independence btw features
sb.heatmap(titanic_dmy.corr()) #For colours close to 1 or -1 shows dependence. In this case, it will be cream and black
###Drop PClass & Fare coz they are dependent
titanic_dmy.drop(['Fare', 'Pclass'], axis=1, inplace=True)
titanic_dmy.head()
##Check to see if dataset size is sufficient to do Logistic regression
##We should have at least 50 records/datapoint per predictive feature. See line 93
   #In this case we have 6 features. That would mean we need 300 datapoints in the dataset
   #We have enough data (889)

###Model Deployment
#I need to split my dataset into training and test dataset
#I am setting aside 80% for training set and 20% for test set.
#y(predictant) = Survived, x(predictors) = everyother column asides 'Survived'
x_train, x_test, y_train, y_test = train_test_split(titanic_dmy.drop('Survived', axis=1),
                                                    titanic_dmy['Survived'], test_size=0.2, random_state=200)
print(x_train.shape)
print(y_train.shape)
x_train[0:5]

#Instantiate the Logistic Regression model
LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(x_train, y_train)

y_pred = LogReg.predict(x_test)

#Calculating the accuracy of the model
print("The accuracy of the model is:", str(round((accuracy_score(y_test, 
                                             y_pred)), 3)) + '%\n')

##Report without cross-validation
print(classification_report(y_test, y_pred))  #This prints the report of the classification
print(confusion_matrix(y_test, y_pred))  #This prints the confusion matrix (This is the performance measurement for ML classification)
##Report with cross-validation
y_train_pred = cross_val_predict(LogReg, x_train, y_train, cv=5)
print(confusion_matrix(y_train, y_train_pred)) #377(TP) and 180(TN) are the numbers of correct predictions

precision_score(y_train, y_train_pred)

######Making a test prediction
titanic_dmy[863:864] #To see what the passenger details look like
## keeping the passengerID, Tweaking the age and removing 'Survived' from details for prediction
test_passenger = np.array([866, 40, 0, 0, 0, 0, 0, 1]).reshape(1,-1)

print(LogReg.predict(test_passenger)) #it returns '1' which is correct bcoz the passengerID(866) actually survived.
#In theory, it means this two guys will survive
print(LogReg.predict_proba(test_passenger)) #The probability of this prediction being correct is 73.6%



# %%
