#%%
import pandas as pd

#File name including extension
filename = "Andy_Punch_20_throws.txt"
#Path to the text file
path = "C:/Users/Zack/Desktop/"
#Read in the file
df = pd.read_csv(path + filename, sep = "\t", skiprows = [0])

#New names of the columns
column_names = ['address', 'Time_s', 'ChipTime', 'ax_g', 'ay_g', 'az_g',
                'wx_deg_s', 'wy_deg_s', 'wz_deg_s', 'AngleX_deg',
                'AngleY_deg', 'AngleZ_deg', 'T_deg', 'hx', 'hy', 'hz', 'D0',
                'D1', 'D2', 'D3', 'Pressure_Pa', 'Altitude_m', 'Lon_deg',
                'Lat_deg', 'GPSHeight_m', 'GPSYaw_deg', 'GPSV_km_h', 'q0',
                'q1', 'q2', 'q3', 'SV', 'PDOP', 'HDOP', 'VDOP']
#Sets the names of the columns
df.columns = column_names

#Every other row contains na's so we delete them
df = df.dropna(axis = 'rows')
#Resets the index, just in case this is used later
df = df.reset_index(drop = True)

#Print
df.head()

#Saves the file to the same location with the same name, different extension
df.to_csv(path + filename.split(".")[0] + ".csv", index = False)
