"""
                        PAMAP2 COMBINING SCRIPT
    This script takes a cleaned file and combines it with all of the other
    clean files in the 'cleaned' directory. If the file does not yet have the
    'subject_id' column, it creates it with information from the name of the
    file.

    INSTRUCTIONS
    
    Place this script in a folder. Create a folders in the same directory and
    name it 'cleaned'. The files must be contained within the 'cleaned' folder.
    The .csv file names should be subjectNUMBER.csv with NUMBER being an integer.
    
    Below all of the functions is a variable called 'path'. Change the 'path'
    variable to the path that this script is contained within. Make sure to
    include the '/' at the end. The 'final_name' variable will be the name of
    the combined .csv file. It will be saved in the same directory as this .py
    script.
    
    Directory Structure:
    C:/Path/Cake/raw/
    C:/Path/Cake/cleaned/
    C:/Path/Cake/pamap2_combine_csv.py    
"""

import pandas as pd
import os
import numpy as np

#------------------------------------------------------------------------------
#START FUNCTIONS

#Returns a list of all of the .csv filenames in a directory
def GetFilenames(p):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))    
    return files

#Returns a list of all of the filenames in the 'cleaned' directory
def GetCleanedFilenames(p):
    return GetFilenames(p + "cleaned/")

#Gets the info from the filename
def FileInfo(file):
    #Gets the actual file name and cuts off the extension
    n = file.split("/")[-1].split(".")[0]
    
    #Gets the person's id, name is always subjectNUMBER, so i split by t
    subject_id = int(n.split("t")[1])
    
    #Returns the info from the filename
    return (subject_id)

#Adds the info from the file names to the dataframe's columns
def CreateInfoColumns(file):
    #Reads the file information from the file name
    subject_id = FileInfo(file)

    #Reads in the cleaned .csv file
    df = pd.read_csv(file)
    
    #If the file already has these columns we skip the adding of these
    if 'subject_id' in df.columns:
        return df
    
    #Saves the number of rows in the dataframe
    num_rows = len(df.index)
    #Creates a list with the desired value duplicated 'num_rows' times
    subj_id = [subject_id]*num_rows

    #Adds the new data lists as a column to the dataframe    
    df["subject_id"] = np.asarray(subj_id)
    
    return df

def CombineAndSaveFiles(list_df, path, final_name):
    #Combines all of the dataframes into a sile dataframe
    df_combined = pd.concat(list_df)
    
    #Saves the combined dataframe to a .csv file
    df_combined.to_csv(path + final_name, index = False)

def CombineSaveAppendInfoCSVFiles(path, final_name):
    filenames = GetCleanedFilenames(path)
    
    all_df = []
    for file in filenames:
        all_df.append( CreateInfoColumns(file) )
    
    CombineAndSaveFiles(all_df, path, final_name)

#------------------------------------------------------------------------------

#Path to this .csv files to combine
path = "C:/githubrepo/CapstoneA/data/PAMAP2_Dataset/Optional/"
#The name for the combined .csv file
final_name = "ComboPlatter.csv"

#Combines the text files
CombineSaveAppendInfoCSVFiles(path, final_name)
