import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import seaborn as sns
import os
sns.set()

os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")

df = pd.read_csv("ComboPlatter.csv") 
idx = np.r_[8:20, 27:39,46:58]
df.drop(df.columns[idx], axis = 1, inplace=True) 

x_train = df.sample(frac = 0.8, random_state = 0)
x_test = df.drop(x_train.index)
y_train = x_train.pop('exercise_id')
y_test = x_test.pop('exercise_id')

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

