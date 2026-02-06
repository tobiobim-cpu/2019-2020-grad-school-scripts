from os import access
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.core import series

series_obj = Series(np.arange(8), index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6', 'row 7', 'row 8']) ##np.arrange() func creates an array of numbers from 0-7 and index 'row 1' -> 0, 'row 2' -> 1, and 'row 3' -> 2
#print(series_obj)
print(series_obj['row 7']) #returns the corresponding value for 'row 7'
print(series_obj[[0, 7]])  #returns the 1st and the 8th array in series_obj dataframe

#creating a dateframe object with random number generator
np.random.seed(25) #So that it gives you the same number everything you run the random number generator
df_obj = DataFrame(np.random.rand(36).reshape((6, 6)), #creates a dataframe from 36 randomly generated numbers and reshapes it to a 6x6 matrix
                index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6'], ##This gives names to the rows
                columns=['column 1', 'column 2', 'column 3', 'column 4', 'column 5', 'column 6']) ##This gives names to the columns
print(df_obj)

#using .loc[] index to locate values in an array
print(df_obj.loc[['row 2', 'row 5'], ['column 5', 'column 2']]) #The .loc[] function locates the values in those rows and columns

#Data Slicing
ds = series_obj['row 3': 'row 7'] #slices the the dateframe from row 3-7. slices the the dateframe and returns the index values
print(ds)

#comparing with scalars
print(str(df_obj < 0.2),  '\n')  #returns 'True' for numbers less than 0.2 -> Line 27, 28, and 29 returns the same answer!!
#print((df_obj < 0.2),  '\n')
#print(df_obj < 0.2,  '\n'

######Filtering with scalars#######
fs = series_obj[series_obj > 6] ##This returns the dataframe/matrix from the table for the values greater than 6
print(series_obj[series_obj > 6]) #returns the array where the column value is greater than '6'
sf = series_obj > 6
print(sf) ##This returns the boolean values for values greater than 6 from the dataframe. Similar to line 31

#setting values with scalars
series_obj['row 1', 'row 5', 'row 8'] = 8 #setting the values of those rows to '8'
#print(str(series_obj) + '\n')

########treating missing values (USe approximation rather than dropping the variable)########
missing = np.nan
seriess_obj = Series(['row 1', 'row 2', missing, 'row 4', 'row 5', 'row 6', missing, 'row 8'])
#print(str(seriess_obj) + "\n")

#Isnull method
IsN = seriess_obj.isnull()
#print(IsN) ##returns a boolen value for the missing numbers

#filling for missing values
np.random.seed(25)
dff_obj = DataFrame(np.random.rand(36).reshape((6, 6)))
dff_obj.loc[3:5, 0] = missing #rows 3-5 under column 0 would be NaN
dff_obj.loc[1:4, 5] = missing #rows 1-4 under column 5 would be NaN
#dff_obj.loc['3:5', 0] = missing #This can also be used as append
#dff_obj.loc['1:4', 0] = missing #This can also be used as append
#use .iloc[] when you have labels as rows and column(see line 13-15). Use .loc[] for intergers(see line 47 output)
print(dff_obj)
#filled_df = dff_obj.fillna(0) #fill missing values with 0s
#print(filled_df)
#filled_df = dff_obj.fillna({0: 0.1, 5: 1.25}) #fill missing values in column index [0] with 0.1 and [5] with 1.25
#print(filled_df)

#fill_df = dff_obj.fillna(method='ffill') #fill the missing values with the last non-zero number in the column
#print(fill_df)

#counting missing values
sum_missing =dff_obj.isnull().sum() #Sums the number missing values in each column
print(sum_missing)

#filtering out missing values for rows
No_NaN = dff_obj.dropna()
print(No_NaN)

#filtering out missing values for columns
No_NaN = dff_obj.dropna(axis=1)
print(str(No_NaN) + '\n')

#######removing duplicates using .duplicated()########
df_object = DataFrame({'column 1':[1,1,2,2,3,3,3],
                       'column 2':['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                       'column 3':['A', 'A', 'B', 'B', 'C', 'C', 'C']})
print(str(df_object) + '\n')
df_object.duplicated()
#line 0 returned false bcoz there was no line before it while line 4 retuned false because it was the first line with those values before line 5&6
print(df_object.duplicated()) 
print(df_object.drop_duplicates())#dropping the duplicates by rows

df_objects = DataFrame({'column 1':[1,1,2,2,3,3,3],
                       'column 2':['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                       'column 3':['A', 'A', 'B', 'B', 'C', 'D', 'C']})
print(df_objects.drop_duplicates(['column 3'])) #drops dulicates in column 3

######Concatenating and transforming######
DF_obj = pd.DataFrame(np.arange(36).reshape(6,6)) #I believe the 'pd' is redundant
print(DF_obj)

DF_obj_2 = pd.DataFrame(np.arange(15).reshape(5,3)) #creating another DF to practice concatenation
print(DF_obj_2)

print(pd.concat([DF_obj, DF_obj_2], axis=1)) #concatenate based on adding columns. It was done based on the row-index values that's why "NaN" is showing for row 5
print(pd.concat([DF_obj, DF_obj_2])) #concatenate based on adding rows. It was done based on the coloumn-index values that's why "NaN" is showing

#Transforming data by dropping data
print(DF_obj.drop([0,2])) #dropping row '0' and '2'
print(DF_obj.drop([0,2], axis=1)) #dropping column '0' and '2'

#Adding data
series_OBJ = Series(np.arange(6))
series_OBJ.name = 'added variable' #the DF we created will be called 'added variable'
print(series_OBJ)

#joining data using .join()
variable_added = DataFrame.join(DF_obj, series_OBJ)
print(variable_added)

#joining data using .append()
added_datatable = variable_added.append(variable_added, ignore_index=False) #the 'ignore_index =False' is to retain the original index value after appending otherwise use 'TRUE'
print(added_datatable)

#Sorting data
DF_sorted = DF_obj.sort_values(by=(5), ascending=[False]) # sort by the row values(5) in descending order
print(DF_sorted)

#######Grouping and Aggregation#######
cars = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt_1\Exercise Files\Data\mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(cars.head())

cars_groups = cars.groupby(cars['cyl']) #grouping cars by cylinder
print(cars_groups.mean())

cars_groups = cars.groupby(cars['am']) #grouping cars by 'am'
print(cars_groups.mean())












