import numpy as np
import pandas as pd
from matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import seaborn as sns
import os
sns.set()

os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")

# def square(x):
# 	return(float(x) ** 2)

# def norm(input, 13):
# 	x, y, z = (input[i] for i in 13)
# 	return([square(x[i]) + square(y[i]) + square(z[i]) for i in range(0, len(x))])



df = pd.read_csv("ComboPlatter.csv") 
idx = np.r_[8:20, 27:39,46:58,60:62]
df.drop(df.columns[idx], axis = 1, inplace=True) 


accel_list = ['sID1_AccX_g','sID1_AccY_g','sID1_AccZ_g']
# df.accelVector1 = norm(accel_list)

x_train = df.sample(frac = 0.8, random_state = 0)
x_test = df.drop(x_train.index)
y_train = x_train.pop('exercise_id')
y_test = x_test.pop('exercise_id')

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# plt.figure(figsize=(11,7))
# colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
#           '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

# for i, r in enumerate([1,2,3,7,8,9]):
#     plt.subplot(3,2,i+1)
#     plt.plot(x_train[r][:100], label=[y_train[r]], color=colors[i], linewidth=2)
#     plt.xlabel('Samples @100Hz')
#     plt.legend(loc='upper left')
#     plt.tight_layout()