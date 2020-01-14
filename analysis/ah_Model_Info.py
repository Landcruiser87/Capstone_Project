import pandas as pd
import os
import numpy as np
import json
import pickle
import os
os.chdir("C:/githubrepo/CapstoneA/")
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
#from analysis.zg_layer_generator_01 import Layer_Generator

import warnings
warnings.filterwarnings("ignore")

class Model_Info:

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
	def GetRawFilenames(self, path):
		return self.GetFilenames(path)
	
	#Function to clean the pull revelevant model infos
	def PullAccuracies(self, path = "data/test_dir/"):
		#Gets all of the path + filenames + extension
		files = self.GetRawFilenames(path)
		
		#print("###########################################################")
		print(str(len(files)) + " models to pull accuracies from.")
	
		AccDict = []
		tempDict = []
	
		for file, num in zip(files, np.arange(len(files))):
			print(num, end = "...")
			tempDict = self.MakeAccuracyDict(file)
			AccDict.append(tempDict)
	
		print("\nModel Structures have been uploaded")
		#print("###########################################################")
	    
	    #Makes a list of all the unique model types
		unique_model_types = []
		the_list = {}
		for item in AccDict:
			for key, value in item.items():
				if key == "model_type":
					if value not in unique_model_types:
						unique_model_types.append(value)
						the_list[value] = []
					del item["model_type"]
					the_list[value] = the_list[value] + [item]
					break
	    
		#print(the_list)
		return the_list#AccDict
	
	def MakeAccuracyDict(file):
		#Opens and reads JSON
		with open(file,"r") as f:
			data = f.read()
		d = json.loads(data)
		#Extracts Model Info
		model_index = d['hyperparameters']['values']['model_structures_index']
		model_acc = d['metrics']['metrics']['accuracy']['observations'][0]['value']
		model_loss = d['metrics']['metrics']['loss']['observations'][0]['value']
		model_val_loss = d['metrics']['metrics']['val_loss']['observations'][0]['value']
		model_val_acc = d['metrics']['metrics']['val_accuracy']['observations'][0]['value']
		model_type = list(d['hyperparameters']['values'].items())[1][0]
		model_type = model_type.split('_')[0]
		
	
		#Finds the pickle file in that directory and extracts the model structure
		path_pkl = "data/"
		#Loop through data directory
		#if it finds the pickle file that begins with GRU, LSTM, etc
		#Extracts and loads it into a dict
		for i in os.listdir(path_pkl):
			if os.path.isfile(os.path.join(path_pkl, i)) and i.startswith(model_type, 0, len(model_type)):
				with open(os.path.join(path_pkl, i), "rb") as fp:
					model_struct = pickle.load(fp)
					model_struct = model_struct[model_index]
					fp.close()
		
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

#------------------------------------------------------------------------------
#An example of how to pull the info out of the returned dictionary
#Run JSON/pickle extraction
#mi = Model_Info()
#acc_dict = mi.PullAccuracies()

#Prints out the model info
#for key, value in acc_dict.items():
#	print(key)
#	for i in value:
#		print(i)
#------------------------------------------------------------------------------


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
# https://stackoverflow.com/questions/51788550/parsing-json-nested-dictionary-using-python

