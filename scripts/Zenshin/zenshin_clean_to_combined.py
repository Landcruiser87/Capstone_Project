import pandas as pd
import os

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

def CombineAndSaveFiles(list_df, path, final_name):
    #Combines all of the dataframes into a sile dataframe
    df_combined = pd.concat(list_df)
    
    #Saves the combined dataframe to a .csv file
    df_combined.to_csv(path + final_name, index = False)

def CombineSaveAppendInfoCSVFiles(path, final_name):
    filenames = GetCleanedFilenames(path)
    
    all_df = []
    for file in filenames:
        all_df.append( pd.read_csv(file) )
    
    CombineAndSaveFiles(all_df, path, final_name)

#------------------------------------------------------------------------------

#Path to this script
path = "C:/githubrepo/CapstoneA/data/Zenshin_Data/"
#The name for the combined .csv file
final_name = "ComboPlatter.csv"

#Combines the text files
CombineSaveAppendInfoCSVFiles(path, final_name)
