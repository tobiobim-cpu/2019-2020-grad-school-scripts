#Data_Storytelling: for presentations to organizational decision makers (Makes it easy for the audience)
#Data_Showcasing: for presentation to analysts, scientist, mathematicians, and engineers
#Data_Art: For presentation to activists or to the general public

#%%  #To show the figures on vscode
import matplotlib
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy.random import randn
import seaborn as sb
sb.set_style('whitegrid')

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pylab import rcParams
from pandas.core.indexes.base import InvalidIndexError

x = range(1, 10)
y = [1, 2, 3, 4, 0.5, 4, 3, 2, 1]
#plt.plot(x,y)
#plt.show()
#print(plt.plot(x,y))

cars = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt_1\Exercise Files\Data\mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
#print(cars.head())

######Creating Standard Data Graphics#######
#ploting a line chart from a pandas object
#isolating the MPG label
mpg = cars['mpg']
plt.plot(mpg)
#plt.show()

#Creating Line charts
df = cars[['cyl', 'wt', 'mpg']]
plt.plot(df)
#plt.show()

#Creating Bar Charts from a list
plt.bar(x,y)
#plt.show()

#Creating a bar chart from pandas object
mpg.plot(kind='bar')
#plt.show()
#plot the bar chart horinzontally
mpg.plot(kind='barh')
#plt.show()

#Creating a pie chart and saving the fig
plt.pie(x)
plt.savefig('pie_chart.png')
#plt.show()

#######defining elements of a plot#########
fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])

#adding limits to my x and y axis
ax.set_xlim([1,9])
ax.set_ylim([0,6])
#adding ticks to my x and y axis
ax.set_xticks([0,1,2,4,5,6,8,9,10])
ax.set_yticks([0,1,2,3,4,5])

ax.grid()
ax.plot(x,y)
#plt.show()

#Generating multiple plots in one figure with subplots
fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(x)
ax2.plot(x,y)
#plt.show()
##Trying the subplot function for the cars dataset
fig = plt.figure()
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(cars['mpg'])
ax2.plot(cars['cyl'])
plt.show()

########Plot formatting######
plt.bar(x,y)
wide = [.5,.5,.5,.9,.9,.9,.5,.5,.5]
color = ['salmon']
plt.bar(x, y, width=wide, color=color, align='center')
plt.show()

#Changing the color of plots
df = cars[['cyl', 'mpg', 'wt']]
color_theme = ['darkgray', 'lightsalmon', 'powderblue']
df.plot(color=color_theme)
plt.show()

#changing color based on RGB codes
z = [1,2,3,4,0.5]
color_themes = ['#A9A9A9', '#FFA07A', '#B0E0E6', '#FFE4C4', '#BDB76B']
plt.pie(z, colors=color_themes)
plt.show()

#customizing line styles
x1 = range(0,10)
y1 = [10,9,8,7,6,5,4,3,2,1]
plt.plot(x,y, ds='steps', lw=5)
plt.plot(x1,y1, ls='--', lw=10)
#plt.show()

"""""
#setting plot markers
plt.plot(x,y, marker='1', mew=20)
plt.plot(x1,y1, marker='+', mew=15)
plt.show()


#Labels and Annotations (functional method)
plt.bar(x,y)
plt.xlabel("Your x-axis label")
plt.ylabel("Your y-axis label")

"""""

z = [1,2,3,4,.5]
veh_type = ['bicycle', 'motorbike', 'car', 'van', 'stroller']
plt.pie(z, labels=veh_type)
plt.show()

#Labels and Annotations (object-oriented method)
rcParams['figure.figsize'] = 8,4
fig = plt.figure()
ax.grid()
ax = fig.add_axes([.1,.1,1,1])
mpg.plot()

ax.set_xticks(range(32))
ax.set_xticklabels(cars.car_names, rotation=60, fontsize='small')
ax.set_title("Miles per Gallon of cars in mtcars dataset")
ax.set_xlabel('car names')
ax.set_ylabel('miles/gal')

plt.grid()
plt.show()

#adding legend to the plot
#Functional method
plt.pie(z)
plt.legend(veh_type, loc='best')
plt.show()
#object-oriented method
fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
mpg.plot()

ax.set_xticks(range(32)) #We have 32 cars
ax.set_xticklabels(cars.car_names, rotation=60, fontsize='small')
ax.set_title("Miles per Gallon of cars in mtcars dataset")
ax.set_xlabel('car names')
ax.set_ylabel('miles/gal')
ax.legend(loc='best')
plt.grid(True)
plt.show()

#Annotting the plot
fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
mpg.plot()

ax.set_xticks(range(32)) #We have 32 cars
ax.set_xticklabels(cars.car_names, rotation=60, fontsize='small')
ax.set_title("Miles per Gallon of cars in mtcars dataset")
ax.set_xlabel('car names')
ax.set_ylabel('miles/gal')
ax.legend(loc='best')
ax.set_ylim([0,45]) #setting limit d=for y axes
#mpg.max() -> 33.9(Honda Civic)
#mpg.min() -> 10.4(Merc 4505LC & Cardillac Fleetwood)
ax.annotate('Honda Civic(Highest @ 33.9)', xy=(19,33.9), xytext=(21,35), 
                          arrowprops=dict(facecolor='red', shrink=0.05))
ax.annotate('Merc 4505LC(Lowest @ 10.4)', xy=(14,10.4), xytext=(4,7), 
                          arrowprops=dict(facecolor='blue', shrink=0.00))
ax.annotate('Cardillac Fleetwood(Lowest @ 10.4)', xy=(15,10.4), xytext=(18,7), 
                          arrowprops=dict(facecolor='blue', shrink=0.00))
plt.grid(True)
plt.show()


#Visualing time series data
address = r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt_1\Exercise Files\Data\Superstore-Sales.csv'
df =pd.read_csv(address, index_col='Order Date', encoding='cp1252', parse_dates=True)
df.head()
#print(df.head())
df['Order Quantity'].plot()

df2 = df.sample(n=100, random_state=25, axis=0)
plt.xlabel('order date')
plt.ylabel('Order quantity')
plt.title('Superstore Sales')
df2['Order Quantity'].plot()


#Creating Statistical data grahics
#plt.hist(mpg)
#plt.show()

#Using Seaborn lib
sb.displot(mpg)

#creating scaatter plot
cars.index = cars.car_names
mpg = cars['mpg']
cars.plot(kind='scatter', x='hp', y='mpg', c=['darkblue'], s=150)
plt.show()

#with seaborn
sb.regplot(x='hp', y='mpg', data=cars, scatter=True)
plt.show()

#scatter plot matrix
sb.pairplot(cars)
plt.show()

#using pandas for scatter plt matrix
pd.plotting.scatter_matrix(cars, figsize=[10,10],  s=100)
plt.show()


# %%
