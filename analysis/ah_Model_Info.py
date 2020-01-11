import pandas as pd
import os
import numpy as np
import json

#Returns a list of all of the .json filenames in a directory
def GetFilenames(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'trial.json' in file:
                files.append(os.path.join(r, file))
    return files

#Returns a list of all of the filenames in the 'raw' directory
def GetRawFilenames(path):
    return GetFilenames(path)

#Function to clean the text file
def PullAccuracies(path):
    #Gets all of the path + filenames + extension
    files = GetRawFilenames(path)
    
    #print("###########################################################")
    print(str(len(files)) + " models to pull accuracies from.")
    #print("\t", end = "")
	AccDF = []
    for file, num in zip(files, np.arange(len(files))):
        print(num, end = "...")
        df_file, name = MakeAccuracyDF(file, AccDF)
        df_file.to_csv(path + "cleaned/" + name, sep = ",", index = False)
        del df_file
    
    #print("\n\tCleaning and saving complete.")
    #print("###########################################################")

def MakeAccuracyDF(file):



# read in the json file

path = "C:/githubrepo/CapstoneA/data/test_dir"
PullAccuracies(path)



# with open(path & "trial.json","r") as f:
# 	data = f.read()

# # Input JSON to dictionary
# d = json.loads(data)

# #Input JSON as dataframe
# # df = pd.read_json(r"C:/githubrepo/CapstoneA/data/test_dir/trial.json")

# print("winner")

# C:/githubrepo/CapstoneA/data/test_dir
# Loop through test_dir and pull accuracy and model structure. 
# Append those values to a new dataframe.
# Sort the dataframe by the highest accuracy, or subset the first 10 
# 
# 

