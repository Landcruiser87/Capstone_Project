import pandas as pd
import os
import numpy as np
import json

#Returns a list of all of the .txt filenames in a directory
def GetFilenames(p):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(p):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))
    return files

#Returns a list of all of the filenames in the 'raw' directory
def GetRawFilenames(p):
    return GetFilenames(p + "raw/")

#Function to clean the text file
def CleanCSVFiles(path):
    #Gets all of the path + filenames + extension
    files = GetRawFilenames(path)
    
    #print("###########################################################")
    print(str(len(files)) + " files to clean and save to csv.")
    #print("\t", end = "")
    for file, num in zip(files, np.arange(len(files))):
        print(num, end = "...")
        df_file, name = CleanFile(file)
        df_file.to_csv(path + "cleaned/" + name, sep = ",", index = False)
        del df_file
    
    #print("\n\tCleaning and saving complete.")
    #print("###########################################################")


# read in the json file
with open("C:/githubrepo/CapstoneA/data/test_dir/trial.json","r") as f:
	data = f.read()

# Input JSON to dictionary
d = json.loads(data)

#Input JSON as dataframe
# df = pd.read_json(r"C:/githubrepo/CapstoneA/data/test_dir/trial.json")

print("winner")

# C:/githubrepo/CapstoneA/data/test_dir
# Loop through test_dir and pull accuracy and model structure. 
# Append those values to a new dataframe.
# Sort the dataframe by the highest accuracy, or subset the first 10 
# 
# 

