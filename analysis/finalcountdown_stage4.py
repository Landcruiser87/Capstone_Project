import pandas as pd
import os
import numpy as np
import json
import pickle
import os
os.chdir("C:/githubrepo/CapstoneA/")
#os.chdir("E:/CAPSTONE STAGES/")
#from analysis.zg_layer_generator_01 import Layer_Generator

import warnings
warnings.filterwarnings("ignore")

class Final_Accuracy:
	
	def ___init__(self, name):
		self.name = name

	#Returns a list of all of the .json filenames in a directory
	def GetFilenames(self, path):
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
	def PullAccuracies(self, nested = True, path = "data/test_dir/"):
		#Gets all of the path + filenames + extension
		files = self.GetRawFilenames(path)
		
		#print("###########################################################")
		print(str(len(files)) + " models to pull accuracies from.")
	
		AccDict = []
		tempDict = []
	
		for file, num in zip(files, np.arange(len(files))):
			print(num, end = "...")
			tempDict = self.MakeAccuracyDict(file)
			if tempDict != {}:
				AccDict.append(tempDict)
	
		print("\nModel Structures have been uploaded")
		#print("###########################################################")
	    
	    #Makes the list be nested (if desired) if not it is a list
		#of dictionaries
		if nested == True:
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
			AccDict = the_list

		return AccDict
	
	def MakeAccuracyDict(self, file):
		#Opens and reads JSON
		with open(file,"r") as f:
			data = f.read()
		d = json.loads(data)
		
		#For some reason a few have no metrics, so we return an empty dictionary
		#print("|" + str(d['metrics']['metrics']) + "|")
		if d['metrics']['metrics'] == {}:
			print("NULL results:", d['trial_id'])
			return {}

		#Extracts Model Info
		model_index = d['hyperparameters']['values']['model_structures_index']
		model_acc = d['metrics']['metrics']['accuracy']['observations'][0]['value']
		model_loss = d['metrics']['metrics']['loss']['observations'][0]['value']
		model_val_loss = d['metrics']['metrics']['val_loss']['observations'][0]['value']
		model_val_acc = d['metrics']['metrics']['val_accuracy']['observations'][0]['value']
		model_type = list(d['hyperparameters']['values'].items())[1][0]
		model_type = model_type.split('_')[0]
		model_hyp = d['hyperparameters']['values']
	
		#Finds the model structure from the pickle file	
		param_path = file.split("\\")[0]
		data = []
		with open(param_path + "/data_parameters.pkl", 'rb') as f:
		    data = pickle.load(f)
		#Model structure
		model_struct = data[1][model_index]
		#Data Parameters
		model_data_params = data[0]
		
		tempDict = {
	        		"model_index" : model_index,
	        		"model_type" : model_type,
	        		"model_struct" : model_struct,
					"model_hyp" : model_hyp,
	        		"acc" : model_acc,
	        		"loss" : model_loss,
	        		"val_acc" : model_val_acc,
	        		"val_loss" : model_val_loss,
					"data_params" : model_data_params
	        		}
	    
		return tempDict

	#Returns the best layer structures by layer setup type (GRU, LSTM, etc)
	def Get_Best_Layer_Structure_Types(self, best_x = 15, parent_folder = "test_dir"):
		dict_acc = self.PullAccuracies(nested = True, path = "data/" + parent_folder + "/")
	
		#Iterate through each type
		model_structures_by_type = {}
		for key in dict_acc:
			#If the model styp is not in the dictionary, add it
			if key not in model_structures_by_type.keys():
				model_structures_by_type[key] = []
			model_structures = self.Get_Best_By_Type(dict_acc[key], best_x, parent_folder)
	
			#Appends the new struture to the list in the dictionary
			model_structures_by_type[key] = model_structures_by_type[key] + model_structures
		
		return model_structures_by_type
	
	#Looks at all of the models by each individual type, sorts them, and returns
	#the best X number of them
	def Get_Best_By_Type(self, models, best_x, parent_folder):
		#Sort them by accuracy
		models = sorted(models, key = lambda i: i['val_acc'], reverse = True) 
		
		#This checks to see the number of models that were saved, if it is less
		#than the requested number it prints out a warning and continues
		if len(models) < best_x:
			print("WARNING: only have", len(models), "not the requested", best_x)
			best_x = len(models)
	
		#Gets the best X model structures
		models_bestx = []
		for i in np.arange(best_x):
			models_bestx.append( models[i]["model_struct"] )
		
		return models_bestx


	#Looks at all of the models by each individual type, sorts them, and returns
	#the best X number of them
	def Get_Best_By_Type_With_Hyperparameters(self, models, folder):
		#Sort them by accuracy
		models = sorted(models, key = lambda i: i['val_acc'], reverse = True) 
		
		#Gets the best X model structures
		models_bestx = []
		for i in np.arange(len(models)):
			models_bestx.append( models[i] )
		
		return models_bestx


	def Get_Accuracy(self, folder = "test_dir"):
		dict_acc = self.PullAccuracies(nested = True, path = "data/" + folder + "/")
	
		#Iterate through each type
		model_structures_by_type = {}
		for key in dict_acc:
			#If the model styp is not in the dictionary, add it
			if key not in model_structures_by_type.keys():
				model_structures_by_type[key] = []
			model_structures = self.Get_Best_By_Type_With_Hyperparameters(dict_acc[key], folder)
	
			#Appends the new struture to the list in the dictionary
			model_structures_by_type[key] = model_structures_by_type[key] + model_structures
		
		return model_structures_by_type

	def Restructure_Layer_Hyp(self, best_by_type):
		types_hyp_list = {}
		#Loop through different types (GRU, LSTM, etc)
		for key in best_by_type:
			if key not in types_hyp_list.keys():
				types_hyp_list[key] = {}
			
			all_layers = []
			
			#Loop through the models of that type
			for model in best_by_type[key]:
				index = -1
				
				#These are the hyperparameters for the specific layer
				layer_parameters = {}
	
				#Iterate through the hyperparameters
				for lay_type in model["model_hyp"]:
					#print(lay_type, model["model_hyp"][lay_type])
					#Ignore the model structure index
					if index == -1:
						index = 0
						continue
	
					s = lay_type.split("_", 2)

					#The optimizer only has 1 value in the split	
					if len(s) == 1:
						#This is a new layer, add it + index to dictionary
						if str(s[0]) not in layer_parameters.keys():
							layer_parameters[str(s[0])] = {}
							index += 1
					else:
						#This is a new layer, add it + index to dictionary
						if str(s[0] + "_" + s[1]) not in layer_parameters.keys():
							layer_parameters[str(s[0] + "_" + s[1])] = {}
							index += 1
					
					#Makes the dictionary of layer parameters
					if len(s) == 1:
						hyp_name = s[0]
					else:
						hyp_name = self.Get_Real_Hyp_Name(s[2])
					hyp_val = self.Get_Real_Hyp_Type(hyp_name, model["model_hyp"][lay_type], model["data_params"][0])
					if len(s) == 1:
						layer_parameters[str(s[0])][hyp_name] = [hyp_val]
					else:
						layer_parameters[str(s[0] + "_" + s[1])][hyp_name] = [hyp_val]
				
				layer_parameters["optimizer"] = model["model_hyp"]["optimizer"]
				struct = model["model_struct"]
				acc = model["acc"]
				loss = model["loss"]
				val_acc = model["val_acc"]
				val_loss = model["val_loss"]
				data_params = model["data_params"]
				info = [struct, acc, loss, val_acc, val_loss, list(data_params)]
				#Check to see if this structure is already in the list
				pos = self.Struct_Position(all_layers, struct)
				if pos == -1:
					#This is a new layer structure, add a new entry
					all_layers.append([info, layer_parameters])
				else:
					#Layer structure already exists, add hyperparameter list
					all_layers[pos].append(layer_parameters)
	
			types_hyp_list[key] = all_layers
	
		return types_hyp_list
	
	def Struct_Position(self, all_layers, cur_struct):
		#Checks if this layer setup already exists	
		for i in np.arange(len(all_layers)):
			if all_layers[i][0] == cur_struct:
				return i
	
		#There is no layer, so we reutrn -1	
		return -1
	
	def Is_Int(self, val):
		try:
			int(val)
			return True
		except:
			return False
	
	def Is_Float(self, val):
		try:
			float(val)
			return True
		except:
			return False
	
	def Get_Real_Hyp_Type(self, name, val, window_size = 200.0):
		#Setting the value to an int/float if it is one
		if self.Is_Float(val):
			v = float(val)
			if self.Is_Int(val):
				val = int(val)
				if val != v:    #This makes sure to not make 0.2 -> 0
					val = v
			else:
				val = float(val)
		
		if name == "filters":
			#Our saved filter size is a percentage of window size, converting to that
			val = val/float(window_size)
			#Just in case the value isn't exact, we round it to the a valid value
			val = min([0.25, 0.5, 0.75], key=lambda x:abs(x-val))
		elif name == "activation":
			#LeakyReLU acts weird so this might be necessary
			if val.lower() != "relu" and val.lower() != "tanh":
				val = "LeakyReLU"
	
		return val
	
	def Get_Real_Hyp_Name(self, name):
		params = ["units", "activation", "bias_initializer", "dropout", "filters",
				   "n_steps", "rate", "pool_size"]
	
		#This is the only hyperparameter name that is not exactly itself
		if name not in params:
			name = "activation"
			
		return name

	#Keras-tuner somehow in the hyperparameter tuning stage adds extra unused
	#hyperparameters, so we remove them
	def Remove_Extra_Parameters(self, raw_acc):
		acc = {}

		#Looping by the category
		for key in raw_acc:
			#Looping by all the items in that category
			all_for_this_type = []
			for mod in raw_acc[key]:
				#This is the location of the layers
				layers = mod[0][0]
				#These are the hyperparameters
				hyp = mod[1]
				#Pulling out only the hyperparamters we use 
				real_hyp = {}
				for i in np.arange(len(layers)):
					if layers[i] != "Flatten":
						real_hyp[layers[i] + "_" + str(i)] = mod[1][layers[i] + "_" + str(i)]
				real_hyp["optimizer"] = mod[1]["optimizer"]
				#Adds these parameters and the model info to the list
				all_for_this_type.append([mod[0], real_hyp])
			#Appends all the info+parameters to the dictionary
			acc[key] = all_for_this_type
		
		return acc
	
	#Generates and saves the final results .pkl file
	def Generate_The_File(self, folder = "step4"):
		raw_acc_s4 = self.Get_Accuracy(folder = folder)
		raw2_acc_s4 = self.Restructure_Layer_Hyp(raw_acc_s4)
		acc_s4 = self.Remove_Extra_Parameters(raw2_acc_s4)
		
		with open("results/results.pkl", "wb") as fp:   #Pickling
			pickle.dump(acc_s4, fp)
		
		return
	
	#Loads in the final results .pkl file
	def Load_The_File(self, directory = 'data/step4/'):
		data = []
		with open("results/results.pkl", 'rb') as f:
		    data = pickle.load(f)
			
		return data
		
	#Tells ya stuff
	def Help(self):
		print("""
		Generate_The_File function generates the DICTIONARY file and saves it
		to the indicated folder.
		Load_The_File function loads in the DICTIONARY file and returns it. 
		
		The keys in the DICTIONARY are the Neural Network categories (GRU,
		LSTM, etc). Those keys give you a LIST of the models run for that category.

		for model in acc_file["Conv1D"]:
			print(model)

		model[0] will give you a LIST of:
			layer structure, acc, loss, val_acc, val_loss, data_params
		layer structure is a LIST of the layers
		data_params is a LIST of:
			window_size, overlap_percent, batch_size
		
		model[1] gives you the DICTIONARY containing the hyperparameters. Each
		parameter is of the form TYPE_NUMBER where TYPE is the neural network
		type (like GRU, LSTM, etc) and NUMBER is the layer number. Example:
			Conv1D_1
		One other parameter inside model[1] is the optimizer, which can be
		accessed by model[1]['optimizer'].

		fa = Final_Accuracy()
		fa.Help()
		fa.Generate_The_File()   #Generates the file, not necessary if you have it
		acc = fa.Load_The_File() #Loads the .pkl file from the results directory
		""")
		return

#fa = Final_Accuracy()
#fa.Help()
#fa.Generate_The_File()
#acc = fa.Load_The_File()


#print(acc["Conv1D"][0][1])

#fa = Final_Accuracy()
#raw_acc_s4 = fa.Get_Accuracy(folder = "step4")
#raw2_acc_s4 = fa.Restructure_Layer_Hyp(raw_acc_s4)
#acc_s4 = fa.Remove_Extra_Parameters(raw2_acc_s4)

# 0 Model Structures, 1 Accuracy, 2 Loss,
# 3 Validation Accuracy, 4 Validation Loss, 5 Data Parameters

#Prints out all the accuracies
#for model in acc_s4["Dense"]:
#	print(model[0][3])

#i = 0
#Prints out top 100 accuracies
#for model in acc_s4["Conv1D"]:
#	print(model)
#	i += 1
#	if i > 10:
#		break
	
	
