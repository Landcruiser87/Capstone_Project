import pandas as pd
import os
import numpy as np
import json
import pickle
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

#Returns a list of all of the filenames in the 'test_dir' directory
def GetRawFilenames(path):
	return GetFilenames(path)

#Function to clean the pull revelevant model infos
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
#Example Json extraction.
#Probably won't use  but neat code that i'll save for later for parsing a JSON into a dictionary/list
# def dict_get(x,key,here=None):
#     x = x.copy()
#     if here is None: here = []
#     if x.get(key):  
#         here.append(x.get(key))
#         x.pop(key)
#     else:
#         for i,j in x.items():
#           if  isinstance(x[i],list): dict_get(x[i][0],key,here)
#           if  isinstance(x[i],dict): dict_get(x[i],key,here)
#     return here
	
def before(value, a):
    # Find first part and return slice before it.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    return value[0:pos_a]

def MakeAccuracyDict(file):
	#Opens and reads JSON
	with open(file,"r") as f:
		data = f.read()
	d = json.loads(data)
	#Extracts Models Info
	Model_Index = d['hyperparameters']['values']['model_structures_index']
	Model_Type = list(d['hyperparameters']['values'].items())[1][0]
	Model_Type = Model_Type.split('_')[0]


	#Finds the pickle file in that directory and extracts the model structure
	path_pkl = "C:/githubrepo/CapstoneA/data"
	for i in os.listdir(path_pkl)
		if os.path_pkl.isfile(os.path.join(path_pkl)) and Model_Type in i:
			file_pkl = open(path_pkl & "/" & i)
			Model_Struct = file_pkl[Model_Index]
			file_pkl.close()

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




