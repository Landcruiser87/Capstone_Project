import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import seaborn as sns
import os
sns.set()

os.chdir("C:/githubrepo/CapstoneA/")
from analysis.zg_Load_Data import Load_Data

import warnings
warnings.filterwarnings("ignore")

# def square(x):
# 	return(float(x) ** 2)

# def norm(input, 13):
# 	x, y, z = (input[i] for i in 13)
# 	return([square(x[i]) + square(y[i]) + square(z[i]) for i in range(0, len(x))])

data_params = {'dataset' : 'pamap2',
               'train_p' : 0.8,
               'w_size' : 0,
               'o_percent' : 0.25
               }
dataset = Load_Data(**data_params)

num_neighbors = dataset.y_train.shape[1]

#Undoes the one-hot encoding
y_test = pd.DataFrame(pd.DataFrame(dataset.y_test).idxmax(1))
y_train = pd.DataFrame(pd.DataFrame(dataset.y_train).idxmax(1))

knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean')
knn.fit(dataset.x_train, y_train)
y_pred = knn.predict(dataset.x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Seeing something
# plt.figure(figsize=(11,7))
# colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
#           '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

# for i, r in enumerate([1,2,3,7,8,9]):
#     plt.subplot(3,2,i+1)
#     plt.plot(x_train[r][:100], label=[y_train[r]], color=colors[i], linewidth=2)
#     plt.xlabel('Samples @100Hz')
#     plt.legend(loc='upper left')
#     plt.tight_layout()