
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import pickle

df=pd.read_csv("Crop_recommendation.csv")

X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=1,max_depth=0)



DecisionTree=DecisionTree.fit(X_train,y_train)



pickle.dump(DecisionTree.open('DecisionTree_model.pkl','wb'))

