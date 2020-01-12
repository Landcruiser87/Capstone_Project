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
	#Opens and reads JSON
	with open(file,"r") as f:
		data = f.read()
	d = json.loads(data)
	#Extracts Models Info
	model_index = d['hyperparameters']['values']['model_structures_index']
	model_acc = d['metrics']['metrics']['accuracy']['observations'][0]['value']
	model_loss = d['metrics']['metrics']['loss']['observations'][0]['value']
	model_val_loss = d['metrics']['metrics']['val_loss']['observations'][0]['value']
	model_val_acc = d['metrics']['metrics']['val_accuracy']['observations'][0]['value']
	model_type = list(d['hyperparameters']['values'].items())[1][0]
	model_type = model_type.split('_')[0]
	

	#Finds the pickle file in that directory and extracts the model structure
	path_pkl = "C:/githubrepo/CapstoneA/data"
	#Loop through data directory
	#if it finds the pickle file that begins with GRU, LSTM, etc
	#Extracts and loads it into a dict

	for i in os.listdir(path_pkl):
		if os.path.isfile(os.path.join(path_pkl, i)) and i.startswith(model_type, 0, len(model_type)):
			with open(os.path.join(path_pkl, i), "rb") as fp:
				model_struct = pickle.load(fp)
				model_struct = model_struct[model_index]
				fp.close()
				break
	tempDict = {
		"model_index" : model_index,
		"model_type" : model_type,
		"model_struct" : model_struct,
		"acc" : model_acc,
		"loss" : model_loss,
		"val_acc" : model_val_acc,
		"val_loss" : model_val_loss,
		}
	return tempDict



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




