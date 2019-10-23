  
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
import matplotlib.pyplot as plt
import timeit

#%%
#imports the data into a dataframe
#Exports it to a csv

from datetime import datetime
from scripts.PAMAP2 import preprocessing as builder

#File name including extension
filename = 'Andy_Punch_rnd1_20throws.txt'

#Path to the text file
path = "C:/Users/andyh/OneDrive/Documents/GitHub/Capstone_Project/data/Captured_Data/Punch/"

#Read in the file
df = builder.build_df(path,filename)
builder.df_info(df)

#Sets the time to a datetime
#Not sure we need this actually, but keeping for now
# times = []
# for index, row in df.iterrows():
#     times.append(pd.Timestamp(row['Time_s']))

#create datetime variable
# df['Time_n'] = np.asarray(times)

# #Moves last column to first
# cols = list(df.columns)
# cols = [cols[-1]] + cols[:-1]
# df = df[cols]

#Print top of df
df.head()

#Saves the file to the same location with the same name, different extension
# df.to_csv(path + filename.split(".")[0] + ".csv", index = False)

#%%

#Summary Stats and count of missing values
print("Structure of data:\n",df.shape,"\n")
print("Count of missing values:\n",df.isnull().sum().sort_values(ascending=False),"\n")
print("Summary Statistic's:\n",round(df.describe(),2),"\n")

#%%
# Plot basic punch
builder.plot_accel('Punch Acceleration Data', df)



#%%
# Another look at variable types
for i in df:
    
    print(i, 
    "type: {}".format(df[i].dtype),
    "# unique: {}".format(df[i].nunique()),
    sep="\n  ", end="\n\n")
    


#%%
# Lets work on getting the integral first. 
path = "C:/Users/andyh/OneDrive/Documents/GitHub/Capstone_Project/data/Captured_Data/Punch/"
