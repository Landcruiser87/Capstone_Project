  
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

# Listing activities and subjects
activity_cat = list(df_fitness.activityID.unique())
print(activity_cat)
subject_cat = list(df_fitness.subject_id.unique())
print(subject_cat)

#Looping through headers to get summary stats on type and counts
df_fitness_headers = df_fitness.head()
for i in df_fitness_headers:
    
    print(i, 
    "type: {}".format(df_fitness[i].dtype),
    "# unique: {}".format(df_fitness[i].nunique()),
    sep="\n  ", end="\n\n")



#%%

def process_drops(df, cols):
    return df.drop(cols,axis=1,inplace=True)

# def process_na_drops(df, cols):
#     return df.dropna(cols,axis=1,inplace=True)

# Looking at one subject and one activity(walking)
df_filt = []
df_filt = df_fitness[df_fitness.subject_id==101]
df_filt = df_filt[df_fitness.activityID==4]

#Filling in heart rate data to match the previous value
# df_filt.heart_rate.fillna(method='ffill', inplace=True)

#Dropping Columns with no data
drop_cols = []
drop_cols = [
    'heart_rate',
    'IMU_hand_orient_0',
    'IMU_hand_orient_1',
    'IMU_hand_orient_2',
    'IMU_hand_orient_3',
    'IMU_chest_orient_0',
    'IMU_chest_orient_1',
    'IMU_chest_orient_2',
    'IMU_chest_orient_3',
    'IMU_ankle_orient_0',
    'IMU_ankle_orient_1',
    'IMU_ankle_orient_2',
    'IMU_ankle_orient_3'
    ]
#Dropping columns
process_drops(df_filt, drop_cols)

#Dropping NA values
df_filt.dropna(subset=[
    'IMU_hand_temperature_C',
    'IMU_hand_acc_z_6g',
    'IMU_hand_mag_z_uT',          
    'IMU_hand_mag_y_uT',         
    'IMU_hand_mag_x_uT',          
    'IMU_hand_gyro_z_rad_s',      
    'IMU_hand_gyro_x_rad_s',      
    'IMU_hand_gyro_y_rad_s',      
    'IMU_hand_acc_y_6g',          
    'IMU_hand_acc_x_6g',          
    'IMU_hand_acc_z_16g',         
    'IMU_hand_acc_y_16g',         
    'IMU_hand_acc_x_16g',         
    'IMU_chest_gyro_x_rad_s',     
    'IMU_chest_temperature_C',    
    'IMU_chest_acc_x_16g',      
    'IMU_chest_acc_y_16g',       
    'IMU_chest_acc_z_16g',        
    'IMU_chest_acc_x_6g',        
    'IMU_chest_acc_y_6g',         
    'IMU_chest_acc_z_6g',         
    'IMU_chest_gyro_y_rad_s',     
    'IMU_chest_gyro_z_rad_s',     
    'IMU_chest_mag_x_uT',         
    'IMU_chest_mag_y_uT',         
    'IMU_chest_mag_z_uT',         
    'IMU_ankle_acc_z_6g',         
    'IMU_ankle_mag_y_uT',         
    'IMU_ankle_mag_x_uT',         
    'IMU_ankle_gyro_z_rad_s',     
    'IMU_ankle_gyro_y_rad_s',     
    'IMU_ankle_gyro_x_rad_s',     
    'IMU_ankle_acc_z_16g',        
    'IMU_ankle_acc_y_6g',        
    'IMU_ankle_acc_x_6g',         
    'IMU_ankle_mag_z_uT',         
    'IMU_ankle_acc_y_16g',       
    'IMU_ankle_acc_x_16g',        
    'IMU_ankle_temperature_C'], inplace=True)

print("Summary Statistic's:\n",round(df_filt.describe(),2),"\n")

#%%
print("Count of missing values:\n",df_filt.isnull().sum().sort_values(ascending=False),"\n")

#%%
