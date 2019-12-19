import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import integrate
#import pandas as pd
random.seed(4)

#------------------------------------------------------------------------------
#START OF FUNCTIONS

#val is the value to integrate, t is a 2 value list of time points p and q
def calculate_integral(cal, t):
    return integrate.quad(lambda x:cal, a = t[0], b = t[1])[0]

#Calculates the integration of a value over time
#value is the value to be integrated
#t is time
#initial_vinteg_val is the starting value for the integrated term
#(like if this was integrating acceleration, it would be the starting velocity)
def calculate_integration_over_time_with_initial(value, t, initial_integ_val = 0):
    #Creates the start/end time with the value to integrate on
    val_a = []
    pairs_t = []
    for i in np.arange(len(value)):
        val_a.append(value[i])
        if i == 0:
            continue
        pairs_t.append([t[i-1], t[i]])
    
    #Does the integration over the time periods with the acceleration
    vel = []
    for i in np.arange(len(pairs_t)):
        prev = initial_integ_val
        if i != 0:
            prev = vel[-1]
        vel.append(calculate_integral(val_a[i], pairs_t[i]) + prev)
    
    return vel

#Calculates the velocity from acceleration
#ac is the acceleration to be integrated on
#t is time
#initial_vel is the initial velocity
def calculate_all_velocity(ac, t, initial_vel = 0):
    return calculate_integration_over_time_with_initial(ac, t, initial_vel)

#Calculates the position from velocity
#vel is the velocity to be integrated on
#t is time
#initial_pos is the initial velocity    
def calculate_all_position(vel, t, initial_pos = 0):
    return calculate_all_velocity(vel, t, initial_pos)

#------------------------------------------------------------------------------
#START MAIN

#Creating the fake time and acceleration data
acceleration = [0]
time = [0]
for i in np.arange(90):
    acceleration.append(acceleration[i] + 5)# random.randint(-2,2))
    time.append(i+1)

#------------------------------------------------------------------------------

velocity = calculate_all_velocity(acceleration, time, 10)
position = calculate_all_position(velocity, time)

plt.plot(acceleration)
plt.show()
plt.plot(velocity)
plt.show()
plt.plot(position)
plt.show()