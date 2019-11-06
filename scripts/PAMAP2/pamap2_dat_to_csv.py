"""
                            PAMAP2 DATASET.DAT TO .CSV
    
    Place this script in a folder. Create 2 folders in the same directory and
    name them 'cleaned' and 'raw'. The raw files must be contained within the
    'raw' folder. The raw .dat file names should be subjectNUMBER. Add the name
    of each of them to the 'filenames' list. This program will convert and save
    them to .csv.

    NO CLEANING DONE YET.
    
    Below all of the functions is a variable called 'path'. Change the 'path'
    variable to the path that this script is contained within. Make sure to
    include the '/' at the end.
    
    Directory Structure:
    C:/Path/Cake/raw/
    C:/Path/Cake/cleaned/
    C:/Path/Cake/pamap2_dat_to_csv.py
"""

import pandas as pd

#------------------------------------------------------------------------------
#START FUNCTIONS

#Makes list of the names of the columns
def make_column_names():
    names = ["timestamp_s", "activityID", "heart_rate"]
    categories = ["IMU_hand", "IMU_chest", "IMU_ankle"]
    per_cat = ["temperature_C", "acc_x_16g", "acc_y_16g", "acc_z_16g", "acc_x_6g",
               "acc_y_6g", "acc_z_6g", "gyro_x_rad_s", "gyro_y_rad_s",
               "gyro_z_rad_s", "mag_x_uT", "mag_y_uT", "mag_z_uT", "orient_0",
               "orient_1", "orient_2", "orient_3"]

    #Makes the permutations of categories with per_cat
    #(IMU_hand_temperature_C, IMU_hand_acc_x_16g, etc...)
    cols = names
    for c in categories:
        for pc in per_cat:
            cols.append(str(c + "_" + pc))      #Appends the concatinated name
    
    return cols

#Gets all the listed .dat files and saves them to .csv in the same folder
def read_save(path, filenames):
    #Gets/creates the list of column names
    cols = make_column_names()

    #Iterates through each of the files
    for fn in filenames:
        df = pd.read_csv(path + "raw/" + fn, header = None, sep = " ") #Reads in .dat file
        df.columns = cols                       #Sets the column names
        name = fn.split(".")[0] + ".csv"        #Replaces the .dat with .csv
        df.to_csv(path + "cleaned/" + name, index = False)   #Saves the file
        del df                                  #Removes the dataframe from memory
#------------------------------------------------------------------------------

#Path to this script
path = "C:/githubrepo/CapstoneA/data/PAMAP2_Dataset/Protocol/"
filenames = ["subject101.dat", "subject102.dat", "subject103.dat",
             "subject104.dat", "subject105.dat", "subject106.dat",
             "subject107.dat", "subject108.dat", "subject109.dat"]

read_save(path, filenames)






