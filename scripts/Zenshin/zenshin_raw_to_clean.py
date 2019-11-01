import pandas as pd
import os
import numpy as np

#os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")

#------------------------------------------------------------------------------
#START FUNCTIONS

#Returns a list of all of the .txt filenames in a directory
def GetFilenames(p):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))    
    return files

#Returns a list of all of the filenames in the 'raw' directory
def GetRawFilenames(p):
    return GetFilenames(p + "raw/")

#Creates the new name for the file
def NewFilename_info(file):
    #exerciseid_amtofexercise_sessionid_subjectid
    return file.split("/")[-1]

def File_Info(file):
    #exerciseid_amtofexercise_sessionid_subjectid
    n = file.split("/")[-1].split(".")[0].split("_")

    exercise_id = n[0]
    exercise_amt = n[1]
    session_id = n[2]
    subject_id = n[3]    
    
    return (exercise_id, exercise_amt, session_id, subject_id)

#Cleans the file from Zenshin
def CleanFile(file):
    #Creates the new filename for saving
    new_filename = NewFilename_info(file)
    
    exercise_id, exercise_amt, session_id, subject_id = File_Info(file)
    
    #Read in the file
    df = pd.read_csv(file, sep = ",")
    
    new_col_names = ["Sensor_id", "TimeStamp_s", "FrameNumber", "AccX_g",
                     "AccY_g", "AccZ_g", "GyroX_deg/s", "GyroY_deg/s",
                     "GyroZ_deg/s", "MagX_uT", "MagY_uT", "MagZ_uT",
                     "EulerX_deg", "EulerY_deg", "EulerZ_deg", "QuatW",
                     "QuatX", "QuatY", "QuatZ", "LinAccX_g", "LinAccY_g",
                     "LinAccZ_g", "Pressure_kPa", "Altitude_m",
                     "Temperature_degC", "HeaveMotion_m"]
    
    df.columns = new_col_names
    
    # Delete multiple columns from the dataframe
    df = df.drop(["FrameNumber", "Pressure_kPa", "Altitude_m",
                  "Temperature_degC", "HeaveMotion_m"], axis=1)
    
    df = df.sort_values(['Sensor_id', 'TimeStamp_s'], ascending=[True, True])
    
    #Resets the index, just in case this is used later
    df = df.reset_index(drop = True)
    
    #TODO: COMBINE ALL OF THE TIME SLOT INTO ONE ROW
    df = join_by_timeslot(df)
    
    #If the file already has these columns we skip the adding of these
    if 'exercise_id' in df.columns:
        return (df, new_filename)
    
    #Saves the number of rows in the dataframe
    num_rows = len(df.index)
    #Creates a list with the desired value duplicated 'num_rows' times
    exercise_id_col = [exercise_id]*num_rows
    exercise_amt_col = [exercise_amt]*num_rows
    session_id_col = [session_id]*num_rows
    subject_id_col = [subject_id]*num_rows

    #Adds the new data lists as a column to the dataframe    
    df["exercise_id"] = np.asarray(exercise_id_col)
    df["exercise_amt"] = np.asarray(exercise_amt_col)
    df["session_id"] = np.asarray(session_id_col)
    df["subject_id"] = np.asarray(subject_id_col)
    
    return (df, new_filename)

def join_by_timeslot(df):
    df_sens = []
    df_sens.append(df[df.Sensor_id == 1])
    df_sens.append(df[df.Sensor_id == 2])
    df_sens.append(df[df.Sensor_id == 3])
    
    #RENAME THE COLUMNS FOR JOINING
    for i in np.arange(len(df_sens)):
        df_sens[i] = df_sens[i].drop("Sensor_id", axis=1)
        new_names = []
        for nm in df_sens[i].columns:
            if nm == "TimeStamp_s":
                new_names.append(nm)
            else:
                new_names.append("sID" + str(i+1) + "_" + nm)
        df_sens[i].columns = new_names

    #Inner join x2
    merged_inner = pd.merge(left = df_sens[0], right = df_sens[1],
                            left_on = 'TimeStamp_s', right_on = 'TimeStamp_s')
    merged_inner1 = pd.merge(left = merged_inner, right = df_sens[2],
                            left_on = 'TimeStamp_s', right_on = 'TimeStamp_s')
    return merged_inner1

#Function to clean the text file
def CleanCSVFiles(path):
    #Gets all of the path + filenames + extension
    files = GetRawFilenames(path)
    
    #print("###########################################################")
    #print("\t" + str(len(files)) + " files to clean and save to csv.")
    #print("\t", end = "")
    for file, num in zip(files, np.arange(len(files))):
        #print(num, end = "...")
        df_file, name = CleanFile(file)
        df_file.to_csv(path + "cleaned/" + name, sep = ",", index = False)
        del df_file
    
    #print("\n\tCleaning and saving complete.")
    #print("###########################################################")

#------------------------------------------------------------------------------
#START MAIN PROGRAM

#Path to this script
path = "C:/githubrepo/CapstoneA/data/Zenshin_Data/"

#Cleans and saves the text files
CleanCSVFiles(path)
