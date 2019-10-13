  
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
# Header Import
df_headers = [
    'timestamp','activityID','heartrate',
    'IMU_hand_temp', 
    'IMU_hand_ax1', 'IMU_hand_ay1', 'IMU_hand_az1',  
    'IMU_hand_ax2', 'IMU_hand_ay2', 'IMU_hand_az2',  
    'IMU_hand_rotx', 'IMU_hand_roty', 'IMU_hand_rotz',  
    'IMU_hand_magx', 'IMU_hand_magy', 'IMU_hand_magz',  
    'IMU_hand_oru', 'IMU_hand_orv', 'IMU_hand_orw',  'IMU_hand_orx',  
    'IMU_chest_temp',
    'IMU_chest_ax1', 'IMU_chest_ay1', 'IMU_chest_az1',  
    'IMU_chest_ax2', 'IMU_chest_ay2', 'IMU_chest_az2',  
    'IMU_chest_timerotx', 'IMU_chest_roty', 'IMU_chest_rotz',  
    'IMU_chest_magx', 'IMU_chest_magy', 'IMU_chest_magz',  
    'IMU_chest_oru', 'IMU_chest_orv', 'IMU_chest_orw', 'IMU_chest_orx',  
    'IMU_ankle_temp', 
    'IMU_ankle_ax1', 'IMU_ankle_ay1', 'IMU_ankle_az1',  
    'IMU_ankle_ax2', 'IMU_ankle_ay2', 'IMU_ankle_az2',  
    'IMU_ankle_rotx', 'IMU_ankle_roty', 'IMU_ankle_rotz',  
    'IMU_ankle_magx', 'IMU_ankle_magy', 'IMU_ankle_magz',  
    'IMU_ankle_oru', 'IMU_ankle_orv', 'IMU_ankle_orw',  'IMU_ankle_orx'
]
df_fitness = pd.read_csv("data/PAMAP2_Dataset/Protocol/subject101.dat",
    names=df_headers,
    sep= " ",
)

df_fitness.head(10)

#%%
print("Structure of data:\n",df_fitness.shape,"\n")
print("Count of missing values:\n",df_fitness.isnull().sum().sort_values(ascending=False),"\n")
print("Count of NaN values in HeartRate: " ,df_fitness.loc[df_fitness.heartrate == 'NaN', 'heartrate'].count())


#%%
