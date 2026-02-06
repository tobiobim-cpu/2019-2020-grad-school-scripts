#import packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#import dataset
my_df = pd.read_csv("video_game_data.csv")

#Separate data into input and output objects
X = my_df.drop(["completion_time"], axis=1)
Y = my_df["completion_time"]

#split data into training and test/validation sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#instantiate the model
regressor = RandomForestRegressor(random_state=42)

#Train the model
regressor.fit(x_train, y_train)

#Assess the model accuracy
y_pred = regressor.predict(x_test)

#To view the data side by side
prediction_comparison = pd.DataFrame({"actual": y_test, "Prediction": y_pred})

#R2_score to assess acccuracy
r2_score(y_test, y_pred)
