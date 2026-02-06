import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy
import sklearn

from pandas import Series, DataFrame
from scipy import stats
from scipy.stats.stats import pearsonr, percentileofscore, spearmanr
from pylab import rcParams
from scipy.stats import chi2_contingency
from sklearn import preprocessing
from sklearn.preprocessing import scale

#%%
address = r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt_1\Exercise Files\Data\mtcars.csv'
cars = pd.read_csv(address)
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(cars.describe()) #returns all of the descriptive statistics of the dataset

##Summary Stat
print(cars.sum()) # for columns
print(cars.sum(axis=1)) #for rows
print(cars.median()) #find the median for each variables
print(cars.mean()) #find the mean for each variables.

mpg = cars.mpg #isolate the variable 'mpg' from dataset 'cars'
plt.plot(mpg)
plt.show()
print('The index no of the row with max value is:', mpg.idxmax()) # returns the index number of the row with the max value in MPG

##Variable distribution
print(cars.std()) #returns standard deviation for each of the variables #Measure of how dispersed the data is in relation to the mean.
print(cars.var()) #returns varaiance for each of the variables #Measures the spread btw numbers in a dataset

gear = cars.gear
print(gear.value_counts())
cyl = cars.cyl
print(cyl.value_counts())
cars_cat = cars[['cyl', 'vs', 'am', 'gear', 'carb']]
cyl_group = cars_cat.groupby('cyl')
print(cyl_group.describe())
cars['group'] = pd.Series(cars.cyl, dtype="category")
cars['group'].dtypes
print(cars['group'].value_counts())
ct =pd.crosstab(cars['vs'], cars['cyl'])
print(ct)



##Summarizing categorical variables
cars.index = cars.car_names
print(cars.head(15))

carb = cars.carb
print(carb.value_counts()) #count the unique variables in carb

#groupby method
cars_cat = cars[['cyl', 'vs', 'am', 'gear', 'carb']]
gears_group = cars_cat.groupby('gear')
print(gears_group.describe())

#transform variables to categorical data typt
cars['group'] = pd.Series(cars.gear, dtype="category") ##created another coulumn 'group' & mapped the original values of 'cars.gear' to it.
cars['group'].dtypes
#print(cars.head())
print(cars['gear'].value_counts())
print(cars['group'].value_counts()) ##Line 68 and 69 gives you the same results

#creating crosstabs for categorical data
ct =pd.crosstab(cars['am'], cars['gear'])
print(ct)

##Parametric Correlation Analysis (PCA-To find correlation btw linear related continuous numerical var)
#Correlation does not imply causation
# R = 1, 0, -1 -> +ve, no, -ve correlation
#print(sb.pairplot(cars))
x = cars[['mpg', 'hp', 'qsec', 'wt', 'cyl', 'vs', 'am', 'gear']]
y = cars[['mpg', 'hp']]
#sb.pairplot(x)

mpg = cars['mpg']
hp = cars['hp']
qsec = cars['qsec']
wt = cars['wt']
cyl =cars['cyl']
vs = cars['vs']
am = cars['am']
ger = cars['gear']
#print(qsec.mean(), '\n')




pearsonr_cofficient, p_value = pearsonr(mpg, hp) #To see if there's a linear relationship btw two quantitative variables
print("The pearson correlation coefficient is:", (pearsonr_cofficient)) ## -0.776 -> Strong Negative correlation

##Using pandas to calculate the pearson correlation coefficient (PCC)
corr =x.corr() 
print(corr) ##it gives the summary of the correlation values in "x"

##Using seaborn to visualize the (PCC)
#hm = sb.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
#print(hm)

##Non-parametric correlation analysis
##To find correlation btw categorical, nonlineraly related, non-normally distributed variables
## Spearman rank
##### Variable are ordinal; numeric, but able to be ranked like cate variable
##### variables are related nonlinearly
##### Data is non-normally distributed.
spearmanr_coefficient, p_value = spearmanr(cyl, vs) # is a non-parametric test that is used to measure the degree of association between two variables

print("Spearman rank correlation coefficient is:", spearmanr_coefficient)

##Chi-square
## If p_value is greater than 0.05 -> variables are independent of each other (Accept Null hypothesis)
## if p_value is less than 0.05 -> Variables are not independent of each other (Reject Null hypothesis)
####### It means variables are correlated.
table = pd.crosstab(cyl, am)
chi2, p, dof, expected = chi2_contingency(table.values)
print("Chi-square statistic is  %0.3f and p_value is %0.3f" % (chi2, p), '\n')

###Scaling and Transforming dataset distributions
#####T prevent error and misleading statistic
###Scale by Normalization (putting values btw 0 and 1) and by standardization
###Normalization

print(mpg.describe(), '\n')
mpg_matrix = mpg.values.reshape(-1,1) #shapes it as a one column mtrix
##Now instantiate the min-max scalar object -> it transforms it to defined range (0,1)
scaled = preprocessing.MinMaxScaler() ##preprocessing is a 'module' while MinMaxscaler() is a 'function'
                                      #It takes the default range(0,1) of min = 0 and max = 1
scaled_mpg = scaled.fit_transform(mpg_matrix)
plt.plot(scaled_mpg)
plt.show() ##Shows the distribution of the value(x-axis) is the same but the actual value(y-xis) has changed
           ## compare with Line 30
##If we want to scale to an exact range for the feature. for example (0, 10)
scaled = preprocessing.MinMaxScaler(feature_range=(0,10))

##using scale() to scale your features
##Standadization
unstandard_mpg = scale(mpg, axis=0, with_mean=False, with_std=False)  ##Not standadized
               #No mean and std values
plt.plot(unstandard_mpg)
plt.show() ##We get our original var back in an unscaled form -> like LINE 30

standard_mpg = scale(mpg) ##Standadized now! Mean has been centered to zero and the var now has Normal dist.
plt.plot(standard_mpg)
plt.show()

##Extreme value Analysis for outliers (Outlier detection)
### Useful in detecting fraud, equipment failure, Cybersecurity event.
####Univariate Method -> Tukey Boxplots (whiskers -> 1.5(IQR[distance btw lower & upper quartile] = 75% -25%))
#### How to detect it mathemtically(a=Q1-1.5[IQR]; min.value < a, b=Q3+1.5[IQR]; max.value > b)
address = r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt_1\Exercise Files\Data\iris.data.csv'
iris = pd.read_csv(address)
iris.columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']

X = iris.iloc[:,0:4].values ## This represent the features/predictive variable (first 5 colums)
Y = iris.iloc[:,0:4].values  ## Target variable. prints up to first 5 rows
print(iris[0:5], '\n') ## This is the same as -> print(iris.head())

##Identifying outliers from Tukey boxplot
iris.boxplot(return_type='dict')
plt.plot()
plt.show()
##Identifying and Isolating the outliers in 'Sepal Width'
Sepal_Width = X[:,1]
iris_outliersUP = (Sepal_Width > 4) ##Outliers above the upper quartile
iris_outlierLO = (Sepal_Width < 2.05) ##Outliers below the lower quartile
print(iris[iris_outliersUP], '\n') ##Shows the 3-outliers as seen in the figure -> LINE 152
print(iris[iris_outlierLO]) ##Shows the only outlier as seen in the figure -> LINE 152

##Applying Tukey outlier labeling
#pd.options.display.float_format = '(:.1f)'.format
#X_df = pd.DataFrame(X)
#print(X_df.describe())

##Multivariate Analysis for outliers
sb.boxplot(x='Species', y='Sepal Length', data=iris, palette='hls')
 ##Looking at the scatterplot matrix
sb.pairplot(iris, hue='Species', palette='hls')


# %%
