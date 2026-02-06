##K-Means clustering is an unsupervised algorithn where you know how many clusters are appropriatez(Used to predict subgroup within a dataset)
       #Used for predicting groups from within an unlabeled dataset. Predictions are based on the No of centroids present(K) and the nearest mean values, given an Euclidean distance measurement btw observations
       #Used cases: 1)Market price and cost modeling 2)Insurance Claim Fruad Detection 3)Hedge Fund Classification 4)Customer Segmentation
    #Things to Keep in mind:
        #(1)Scale your variables 2)Look at a scatterplot or the data table to estimate the appropriate number of centroids to use for the K-parameter values

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as sm


from pylab import rcParams
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

plt.figure(figsize=(7,4))

##Solving the iris dataset question using KMeans(unsupervised)
#iris = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\iris.data.csv')
iris = datasets.load_iris() #Loading data

x = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
print(x[0:10])

##Building and running the model
clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(x)
##Plotting the model outputs
iris_df = pd.DataFrame(iris.data) ##"pd.Dataframe() is dataframe constructor"
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']
color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])
plt.subplot(1,2,1) #creating a subplot with 1 row and 2 columns. the 3rd parameter indicates the position we want the plot to come
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50) #s = size
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50) #s = size
plt.title('KMeans Classification')
plt.show() ##The model did a great job predicting the models themselves but the labelling is off

##Relabeling
relabel = np.choose(clustering.labels_, [2,0,1]).astype(np.int64) #[2,0,1] is the list of new labels. It is going to assign the label numbers to the clustering labels that as been predictred by our model

plt.subplot(1,2,1) #creating a subplot with 1 row and 2 columns. the 3rd parameter indicates the position we want the plot to come
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50) #s = size
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[relabel], s=50) #change the 'clustering.labels_' to 'relabel'
plt.title('KMeans Classification')
plt.show() #The model has been relabeled

##Evaluate the clustering results
print(classification_report(y, relabel))
#(1) Precision is a measure of the model's relevancy (Always go for high precision)
#(2) Recall is a measure of the model's completeness (Always go for high recall)


##Hierarchical Clustering(Unsupervised learning)
   #It predicts subgroups within data by finding the distance btw each data point and its nearest neighbours, and then linking the most nearby neighbors
   #It uses the distance metric it calculates to predict subgroups.
       #To guess the No of subgroups in a dataset, first look at a dendrogram visulaization of the clustering results
       #Dendrogram: a tree graph that's useful for visually displaying taxonomies, lineages, and relatedness.
       #Used cases (1) Hospital Resource Management (2) Business Process Management 3)Customer segmentation and 4)Social Network analysis
    #Parameters (Use trial and method to see which one fits):
       #Distance Metrics:
          #Euclidean: works with 'average and ward'
          #Manhattan: works with average
          #Cosine
       #Linkage Parameters:
          #Ward
          #Complete
          #Average

from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import seaborn as sb
sb.set_style('whitegrid') #Setting the style for seaborn

np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(5,3))

cars = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

x = cars[['mpg', 'disp', 'hp', 'wt']].values #coz we only want the values
y = cars.iloc[:,(9)].values #I only want the values at position 9

##Using SCIPY to generate dendrograms
z = linkage(x, 'ward') #linkage() carries the hierarchical clustering of the data
dendrogram(z, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=15, show_contracted=True)
plt.title('Truncated Hierarchial Clustering Diagram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()

##Generating hierarchical clusters
k = 2
#Euclidean and ward
hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward') #ward can only work with euclidean
hclustering.fit(x)
sm.accuracy_score(y, hclustering.labels_) #0.78

#Euclidean and average
hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average') #ward can only work with euclidean
hclustering.fit(x)
sm.accuracy_score(y, hclustering.labels_) #0.78

#Manhattan and average
hclustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average') #ward can only work with euclidean
hclustering.fit(x)
sm.accuracy_score(y, hclustering.labels_) #0.718

##*****Read more about Hierarchical clustering*******


##DBSCAN(Unsupervised)
   #It clusters core samples(dense area of a dataset) and denotes non-core samples(sparse portions of the dataset)
   #Used cases: Computer vision project for the advancement of self-driving cars to predict lanes based on the density of the lines(line data will be provided)
     #Used to identify collective outliers. Outliers should <= 5% of total observations
    #Model Parameters:
      #eps: max distance btw 2 samples for them to be clustered within the same neighbourhood. Start value @ 0.1
      #min_samples: minimum number of samples for a datapoint to qualify as a core point. start with a very low sample size

 
from sklearn.cluster import DBSCAN
from collections import Counter

rcParams['figure.figsize'] = 5, 4 #runtime configuration: Each time Matplotlib loads, it defines rc containing the default styles for every plot element you create.

DF = pd.read_csv(r'C:\Users\dobimuyiwa\Documents\Exercise Files\Ex_Files_Python_Data_Science_EssT_Pt2\Exercise Files\Data\iris.data.csv', header=None, sep=',')
DF.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Species']

data = DF.iloc[:,0:4].values #values in the first 4 coloumns
target  = DF.iloc[:,4].values #to select the column at index postion 4

#Instantiate themodel
model = DBSCAN(eps=0.8, min_samples=19).fit(data)
#print(model)
#Visualize your results tonidentify outliers
outliers_df = pd.DataFrame(data)
print(Counter(model.labels_))
print(outliers_df[model.labels_ == -1])

fig = plt.figure()
ax = fig.add_axes([.1,.1,1,1])
colors = model.labels_

ax.scatter(data[:,2], data[:,1], c=colors, s=120)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
plt.title('DBSCAN for outlier Detection')
plt.show()


# %%
