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


import matplotlib.pyplot as plt
import numpy as np 
import scipy.integrate as sint

def trapzl(y, x):
    "Pure python version of trapezoid rule."
    s = 0
    for i in range(1, len(x)):
        s += (x[i]-x[i-1])*(y[i]+y[i-1])
    return s/2

trapzl(df['ax_g'], df['ay_g'])

#%%
