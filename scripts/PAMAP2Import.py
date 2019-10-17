  
#%%
# ms-python.python added
import os
try:
	os.chdir('~/Capstone_Project') 
	print(os.getcwd())
except:
	pass


import pandas as pd
import numpy as np
import seaborn as sns
# import plotly.plotly as py
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
import timeit

# %%
df_fitness = pd.read_csv("data/PAMAP2_Dataset/Protocol/comboplatter.csv",
    sep= ",",
)

df_fitness.head(10)

#%%
print("Structure of data:\n",df_fitness.shape,"\n")
print("Count of missing values:\n",df_fitness.isnull().sum().sort_values(ascending=False),"\n")
print("Count of NaN values in HeartRate: " ,df_fitness.loc[df_fitness.heart_rate == 'Na', 'heart_rate'].count())


#%%

activity_cat = list(df_fitness.activityID.unique())
df_fitness_headers = df_fitness.head()
for i in df_fitness_headers:
    
    print(i, 
    "type: {}".format(df_fitness[i].dtype),
    "# unique: {}".format(df_fitness[i].nunique()),
    sep="\n  ", end="\n\n")
    

#%%
print(df_fitness.head())

