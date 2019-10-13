import pandas as pd
import numpy as np
import seaborn as sns
import timeit
import matplotlib.pyplot as plt

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


def samplerate(n,t):
    sr = int(n/(t.max()-t.min()))
    return sr
