"""
                        WITMOTION CLEANING SCRIPT
    This script takes raw text file output from the sensors, cleans it, then
    saves it into the 'cleaned' folder.
    
    INSTRUCTIONS
    
    Place this script in a folder. Create 2 folders in the same directory and
    name them 'cleaned' and 'raw'. The raw files must be contained within the
    'raw' folder. The raw text file name should be ACTION_PERSONNAME_. Anything
    after the second '_' is ignored.
    
    Below all of the functions is a variable called 'path'. Change the 'path'
    variable to the path that this script is contained within. Make sure to
    include the '/' at the end.
    
    Directory Structure:
    C:/Path/Cake/raw/
    C:/Path/Cake/cleaned/
    C:/Path/Cake/raw_to_clean.py
"""

import pandas as pd
import os
import numpy as np

#------------------------------------------------------------------------------
#START FUNCTIONS

#Returns a list of all of the .txt filenames in a directory
def GetFilenames(p):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))    
    return files

#Returns a list of all of the filenames in the 'raw' directory
def GetRawFilenames(p):
    return GetFilenames(p + "raw/")

#Creates the new name for the file
def NewFilename(file, num):
    #Gets the actual file name
    n = file.split("/")[-1]
    
    #Gets the action type from the file name (we assume it is first in the name)
    action_type = n.split("_")[0].lower()
    
    #Gets the name of the person who performed the action from the file name
    #(we assume it is second position of the filename)
    person_name = n.split("_")[1].lower()
    
    #Returns the name for the newly cleaned file
    return str(action_type + "_" + person_name + "_" + str(num) + ".csv")

#Cleans the file from Witmotion(?)
def CleanFile(file, num):
    #Creates the new filename for saving
    new_filename = NewFilename(file, num)
    
    #Read in the file
    df = pd.read_csv(file, sep = "\t", skiprows = [0])
    
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
    
    return (df, new_filename)

#Function to clean the text file
def CleanTextFiles(path):
    #Gets all of the path + filenames + extension
    files = GetRawFilenames(path)
    
    print("###########################################################")
    print("\t" + str(len(files)) + " files to clean and save to csv.")
    print("\t", end = "")
    for file, num in zip(files, np.arange(len(files))):
        print(num, end = "...")
        df_file, name = CleanFile(file, num)
        df_file.to_csv(path + "cleaned/" + name, index = False)
        del df_file
    
    print("\n\tCleaning and saving complete.")
    print("###########################################################")

#------------------------------------------------------------------------------
#START MAIN PROGRAM

#Path to this script
path = "C:/Users/Zack/Desktop/Raw_to_Awesome/"

#Cleans and saves the text files
CleanTextFiles(path)
