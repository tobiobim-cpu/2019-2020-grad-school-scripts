###Explanatory Factor Analysis(EFA):
   #it is a regression method you apply to a dataset discover root cause/hidden factors that are present in a dataset but not observable
   #used to regress on features in order to discover factors you can use as variables to represent the original dataset
   #These factors are the synthetic rep of your dataset with the extra dimensionality in info redundancy stripped out
   #Factors(latent variables):
      #variables that are quite meaningful but that are inferred and not directly observable
   #Assumptions:
      #Features are metric, continuous or ordinal
      #corr (r) > 0.3 btw features and dataset
      #observations > 100 and > 5 observations per feature
      #sample is homogenous
   #Outputs of factors are called factor loading. ~ -1 or 1 -> factor has strong influence on the variable and 0 -> factor weakly influences on the variables and > 1 -> highly correlated factors

#%%
import numpy as np
import pandas as pd
import sklearn

from sklearn.decomposition import FactorAnalysis
from sklearn import datasets

##Using iris dataset
iris = datasets.load_iris() #Loading data

x = iris.data
variable_names = iris.feature_names

#Instantiate the model
factor = FactorAnalysis().fit(x)
DF = pd.DataFrame(factor.components_, columns=variable_names)
print(DF) #Factor 1 has the highest factor loading on sepal length, petal length, and petal width. Factor 1 is highly influential on these features
          #Factor 2 has no high loading on any of the features. So we can drop the factor for the rest of the analysis

####Principal Component Analysis(PCA, Unsupervised):
  ##Singular Value Decomposition(SVD):
     #A linear algebra method tht decomposes a matrix into three resultant matrices in order to reduce information redundancy and noise
     #SVD is most commonly used for PCA
#PCA is an Unsupervised ML method that discovers the relationship btw variables and reduces variables down to a set of uncorrelated synthetic rep of Principal components
   #You canuse PCA to decompose customer purchasing data into one vector that describes the factors that affects the customer's purchasing behaviour
       #Another vectors that describes the probabilities that products will be purchaed based on those key influencing factors
#PC are a synthetic rep of a dataset. They are features that embody a dataset's important info(like 'variance') with the redundancy, noise and outliers stripped out.
    #Used cases: 1)Fraud detection 2)Spam detection 3)Speech Recognition and 4)image recognition
###Using Factors nd Components:
   #They both rep what is left of a dataset after info redundancy and noise has been stripped out
   #Use them as input variables for ML algorithms to generate predictions from these compressed rep of your data

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
import sklearn

from IPython.display import Image
from IPython.core.display import HTML
from pylab import rcParams
from sklearn import datasets
from sklearn import decomposition
from sklearn.decomposition import PCA

rcParams['figure.figsize'] = 5, 4#dimensions for data visulaization
sb.set_style('whitegrid') #Setting the style for seaborn

##using the previous iris dataset
#Instantiate the model
pca = decomposition.PCA()
iris_pca = pca.fit_transform(x)
pca.explained_variance_ratio_ #useful for seeing how much variance is explained by the component that were found. #Tells us how much info is compressed into the first few components
                              #4rm results, we see that the 1st component explained 92.4% of the dataset variance. That means it holds 92.4% of the data's info in one principal component.
                                  #92.4% in one PC is cool
pca.explained_variance_ratio_.sum() #returns 1.0 -> Tells us that 100% of the dataset info is captured in the 4 components that were returned but we dont want 100% of the info back coz it will have redundant, outliers and noise info
                                    #When deciding how many components to keep, look at %ofCumVariance. Ensure to retain >= 70% of the orignal info of the data

comps = pd.DataFrame(pca.components_, columns=variable_names)
sb.heatmap(comps, cmap='Blues', annot=True)
plt.show()



# %%
