import pandas as pd
import os
import numpy as np
import json
from zg_layer_generator_01 import Layer_Generator

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

	AccDict = []
	tempDict = []

	for file, num in zip(files, np.arange(len(files))):
		print(num, end = "...")
		tempDict = MakeAccuracyDict(file)
		AccDict.append(tempDict)

	print("\nModel Structures have been uploaded")
	#print("###########################################################")
#Example Json extraction.  Only gets the names.  Which... will be useful i guess?
def dict_get(x,key,here=None):
    x = x.copy()
    if here is None: here = []
    if x.get(key):  
        here.append(x.get(key))
        x.pop(key)
    else:
        for i,j in x.items():
          if  isinstance(x[i],list): dict_get(x[i][0],key,here)
          if  isinstance(x[i],dict): dict_get(x[i],key,here)
    return here
def MakeAccuracyDict(file):
	with open(file,"r") as f:
		data = f.read()
	d = json.loads(data)
	Model_Index = d['hyperparameters']['values']['model_structures_index']

	#Model_Index = d['hyperparameters']['space'][0]['config']['name']['default']
	# names = extract_values(d, 'name')
	# print(names)
	#Model_Index = d.Hyperparameters.values['model_structure_index']
	#Parse model type from json
	#String split the folder name in test dir to get structure used
	#Open the appropriate model structure file
	#Load_Model_Structures(name = "GRU_Model_Structures")
	#load pikle file grab index of model structure
	#Append those values to a new dataframe.
	#Sort the dataframe by the highest accuracy, or subset the first 10 
	#Model Structure
	#Hyperparameters
	#Metrics = accuracy val loss etc.  just grab all of it. 
	return d



#Set path and run pull
path = "C:/githubrepo/CapstoneA/data/test_dir"
PullAccuracies(path)



# with open(path & "trial.json","r") as f:
# 	data = f.read()

# # Input JSON to dictionary
# d = json.loads(data)

# #Input JSON as dataframe
# # df = pd.read_json(r"C:/githubrepo/CapstoneA/data/test_dir/trial.json")

# print("winner")

# # C:/githubrepo/CapstoneA/data/test_dir
# # Loop through test_dir and pull accuracy and model structure. 




