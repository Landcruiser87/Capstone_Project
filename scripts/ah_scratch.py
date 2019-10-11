  
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
#get_ipython().run_line_magic('matplotlib', 'inline')

#%%
#imports the data into a dataframe
#Exports it to a csv


import pandas as pd
from datetime import datetime


#File name including extension
filename = 'Andy_Punch_rnd1_20throws.txt'

#New names of the columns
column_names = ['Sensor_id', 'Time_s', 'ChipTime', 'ax_g', 'ay_g', 'az_g',
                'wx_deg_s', 'wy_deg_s', 'wz_deg_s', 'AngleX_deg',
                'AngleY_deg', 'AngleZ_deg', 'T_deg', 'hx', 'hy', 'hz', 'D0',
                'D1', 'D2', 'D3', 'Pressure_Pa', 'Altitude_m', 'Lon_deg',
                'Lat_deg', 'GPSHeight_m', 'GPSYaw_deg', 'GPSV_km_h', 'q0',
                'q1', 'q2', 'q3', 'SV', 'PDOP', 'HDOP', 'VDOP']

#Path to the text file
path = "C:/Users/andyh/OneDrive/Documents/GitHub/Capstone_Project/data/Captured_Data/Punch/"
#Read in the file
df = pd.read_csv(path + filename, header=None, sep = "\t", skiprows = [1], names=column_names)


#Every other row contains na's so we delete them
df = df.dropna(axis = 'rows')
#Resets the index, just in case this is used later
df = df.reset_index(drop = True)

#Sets the time to a datetime

pd.Timestamp("13:44:37.744")
times = []
for index, row in df.iterrows():
    times.append(pd.Timestamp(row['Time_s']))

df['Time_new'] = np.asarray(times)



#Print
df.head()

#Saves the file to the same location with the same name, different extension
df.to_csv(path + filename.split(".")[0] + ".csv", index = False)

#%%

#Summary Stats and count of missing values
print("Structure of data:\n",df.shape,"\n")
print("Count of missing values:\n",df.isnull().sum().sort_values(ascending=False),"\n")
print("Summary Statistic's:\n",round(df.describe(),2),"\n")

#%%
# Another look at variable types
for i in df:
    
    print(i, 
    "type: {}".format(df[i].dtype),
    "# unique: {}".format(df[i].nunique()),
    sep="\n  ", end="\n\n")
    
print("Summary Statistic's:\n",round(df_census.describe().unstack(),2),"\n")


#%%
# Lets work on getting the integral first. 

#%%
#Now that we've fucked around with the data.  Lets build something with it. 
# Lets get a plot of the data 

import matplotlib.pyplot as plt

Time_s = df['Time_s']
x = df['ax_g']
y = df['ay_g']
z = df['az_g']

plt.subplot(3,1,1)
plt.plot(Time_s,x,'.-')
plt.title('X Acceleration')

plt.subplot(3,1,2)
plt.plot(Time_s,y,'.-')
plt.title('Y Acceleration')

plt.subplot(3,1,3)
plt.plot(Time_s,z,'.-')
plt.title('Z Acceleration')
#%%

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np 
import scipy.integrate as sint

def trapzl(y, x):
    "Pure python version of trapezoid rule."
    s = 0
    for i in range(1, len(x)):
        s += (x[i]-x[i-1])*(y[i]+y[i-1])
    return s/2

trapzl(y, x)