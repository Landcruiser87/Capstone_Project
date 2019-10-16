"""
    PAMAP2 DATASET.DAT TO .CSV
    
    Put all the .dat files in the same directory. Add the name of each of them
    to the 'filenames' list. This program will convert and save them to .csv.
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
        df = pd.read_csv(path + fn, header = None, sep = " ") #Reads in .dat file
        df.columns = cols                       #Sets the column names
        name = fn.split(".")[0] + ".csv"        #Replaces the .dat with .csv
        df.to_csv(path + name, index = False)   #Saves the file
        del df                                  #Removes the datafram from memory
#------------------------------------------------------------------------------

path = "C:/githubrepo/CapstoneA/data/PAMAP2_Dataset/Optional/"
filenames = ["subject101.dat"]

read_save(path, filenames)






