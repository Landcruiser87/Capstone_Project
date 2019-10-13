import pandas as pd
import numpy as np
import seaborn as sns
import timeit
import matplotlib.pyplot as plt
import scipy.integrate as sint

##########################
## Dataframe construction
##########################
def build_df(path,filename):

    #WITMotion Column Names
    column_names = ['Sensor_id', 'Time_s', 'ChipTime', 'ax_g', 'ay_g', 'az_g',
                'wx_deg_s', 'wy_deg_s', 'wz_deg_s', 'AngleX_deg',
                'AngleY_deg', 'AngleZ_deg', 'T_deg', 'hx', 'hy', 'hz', 'D0',
                'D1', 'D2', 'D3', 'Pressure_Pa', 'Altitude_m', 'Lon_deg',
                'Lat_deg', 'GPSHeight_m', 'GPSYaw_deg', 'GPSV_km_h', 'q0',
                'q1', 'q2', 'q3', 'SV', 'PDOP', 'HDOP', 'VDOP']

    df = pd.read_csv(path + filename, 
                header=None, 
                sep = "\t", 
                skiprows = [1], 
                names=column_names)

    #Every other row contains na's so we delete them
    df = df.dropna(axis = 'rows')
    #Resets the index, just in case this is used later
    df = df.reset_index(drop = True)

    return df

####################
## Data Specific
####################

def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan

def df_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

def samplerate(n,t):
    sr = int(n/(t.max()-t.min()))
    return sr


####################
## Basic Plots
####################

# Acceleration plot
def plot_accel(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['Time_s'], data['ax_g'], 'X-Axis')
    plot_axis(ax1, data['Time_s'], data['ay_g'], 'Y-Axis')
    plot_axis(ax2, data['Time_s'], data['az_g'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):

    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

