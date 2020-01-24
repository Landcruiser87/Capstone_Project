"""
	1. Layer Generation
	2. Layer Tuning
	3. Hyperparameter Tuning
	4. Data Tuning
	5. Visualize/Analyize
"""

import itertools
import numpy as np
import pandas as pd
import pickle
import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator
from analysis.zg_model_tuning_01 import Model_Tuning
from analysis.ah_Model_Info import Model_Info

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
# 1. Layer Generation - DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Makes an instance of the class
#Generates and saves to disk the huge list of all model layer setups
#Saves and splits up that huge list of all saved layer setups into categories
lay_gen = Layer_Generator()
lay_gen.Generate_Model_Strutures(lay_gen.Generate_Layer_Parameters(), lay_gen.Generate_Layer_Depth())
lay_gen.Split_Save_Models_Into_Categories()

#==============================================================================
#2. Layer Tuning

#The function to call for Layer Tuning, finds the best layer setups
def Find_All_Layers_Accuracy(categories):
	#Loops through each category given
    for layer_type in categories:
        Find_Layer_Accuracy(layer_type)
    
    return

#Runs the Layer Tuning on a specific category
def Find_Layer_Accuracy(layer_type):
    layer_type = layer_type + "_Model_Structures"
    
    #Run tuning on this with the given name
    lay_gen = Layer_Generator(m_tuning = "simple")
    model_structures = lay_gen.Load_Model_Structures(layer_type)
    
	#This is needed because clstm needs some parameters for the data
    clstm_params = {}
    if layer_type == "ConvLSTM2D_Model_Structures":
        clstm_params = lay_gen.Generate_Simple_Layer_Parameters()["ConvLSTM2D"]
    
	#Load the data with the fixed parameters into memory
    data_params = {'dataset' : 'firebusters',
                   'train_p' : 0.8,
                   'w_size' : 200,
                   'o_percent' : 0.25,
				   'LOSO' : True,
                   'clstm_params' : clstm_params
                   }
    dataset = Load_Data(**data_params)

	#Sets up the model tuning class    
    mt = Model_Tuning(model_structures,
                      dataset,
                      m_tuning = layer_type,
					  fldr_name = layer_type,
					  parent_fldr = "step3",
                      fldr_sffx = '1')
	#Runs the model tuner, in this instance it is Layer Tuning
    mt.Tune_Models(epochs = 500, batch_size = 64, MAX_TRIALS = 1000)
    
    return

#A list of all the categories that could be run
#The layers to run on this iteration (not doing all at the same time because its slow)
#Running the layer tuner
"""["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D", "Dense", "GRU", "LSTM"]"""
categories = ["GRU"]
Find_All_Layers_Accuracy(categories)

#==============================================================================
#4. Take BEST 15 for each category (15LSTM, 15GRU...)
#5. Run hyperparameter tuning for BEST of each category

def Run_Hyper_On_Best_By_Category():
	mi = Model_Info()
	parent_folder = "step3"
	best_by_type = mi.Get_Best_Layer_Structure_Types(best_x = 15, parent_folder = parent_folder)
	
	for key in best_by_type:
		Run_Hyperparameter_Tuning(key, best_by_type[key])

def Run_Hyperparameter_Tuning(model_structures_type, model_structures):
	print(model_structures_type)

	layer_type = model_structures_type + "_Models"
    
	lay_gen = Layer_Generator()
	clstm_params = {}
	if model_structures_type == "ConvLSTM2D":
		clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
    
	data_params = {'dataset' : 'firebusters',
                   'train_p' : 0.8,
                   'w_size' : 200,
                   'o_percent' : 0.25,
				   'LOSO' : True,
                   'clstm_params' : clstm_params
                   }
	dataset = Load_Data(**data_params)
    
	mt = Model_Tuning(model_structures,
                      dataset,
                      m_tuning = "all",	 	  #Whether to use simple or all hyperparamters
					  parent_fldr = "step5",   #'Project' folder name
					  fldr_name = layer_type, #This tuning's folder name
                      fldr_sffx = '1')        #Suffix for the folder just in case
	mt.Tune_Models(epochs = 500, batch_size = 64)
    
	return

Run_Hyper_On_Best_By_Category()

#==============================================================================
#6. Pull out TOP 10 for each tuned
#6a. Run the TOP 10 with data hyperparameters

def Run_Data_Hyperparameter_Tuning():
	mi = Model_Info()
	parent_folder = "step3"
	best_by_type = mi.Get_Best_Layer_Structure_Types_With_Hyperparameters(best_x = 10, parent_folder = parent_folder)

	s = mi.Restructure_Layer_Hyp(best_by_type)

	for key in best_by_type:
		Data_Hyperparameter_Tuning(key, best_by_type[key], s[key])

	return
	
def Data_Hyperparameter_Tuning(model_structures_type, model_structures, hyp_str):
	window_size = [400, 200, 50, 10]
	overlap_percent = [50, 25, 0]
	batch_size = [16, 64, 256]

	#Makes all combinations of the three parameters's values
	all_data_parameters = [window_size, overlap_percent, batch_size]
	all_data_parameters = list(itertools.product(*all_data_parameters))
	
	#Used for creating the folders in the tuner
	loop_num = 0
	
	#Save the parameters to the data folder
	with open("data/ModelStr_Hyp.pkl", "wb") as fp:   #Pickling
		pickle.dump(hyp_str, fp)
	
	#Looping through each data parameter set
	for params in all_data_parameters:
		layer_type = model_structures_type + "_Data_Models"
	    
		lay_gen = Layer_Generator()
		clstm_params = {}
		if model_structures_type == "ConvLSTM2D":
			clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
	    
		data_params = {'dataset' : 'firebusters',
	                   'train_p' : 0.8,
	                   'w_size' : params[0],
	                   'o_percent' : params[1],
					   'LOSO' : True,
	                   'clstm_params' : clstm_params
	                   }
		dataset = Load_Data(**data_params)
		   
		mt = Model_Tuning(model_structures,
	                      dataset,
	                      m_tuning = "all",	 	  #Whether to use simple or all hyperparamters
						  parent_fldr = "step6a",   #'Project' folder name
						  fldr_name = layer_type + "_" + model_structures_type + "_", #This tuning's folder name
	                      fldr_sffx = str(loop_num))        #Suffix for the folder
		mt.Tune_Models(epochs = 500, batch_size = params[2])
		
		loop_num += 1

		#Save the data parameters to the file
		curFolder = layer_type + "_" + model_structures_type + "_" + str(loop_num)
		with open("data/step6a/" + curFolder + "/data_parameters.pkl", "wb") as fp:   #Pickling
			pickle.dump(hyp_str, fp)

	#Delete the pkl file
	os.remove("data/ModelStr_Hyp.pkl") 

	return

Run_Data_Hyperparameter_Tuning()

#==============================================================================
#7. Pull out TOP 10 for each tuned
def Get_Da_Best(best_x = 15):
	mi = Model_Info()
	dict_acc = mi.PullAccuracies(nested = True, path = "data/step5/")

	#Prints out the model info (just the accuracy)
	for key, value in dict_acc.items():
		print(key)
		dict_acc[key] = sorted(value, key = lambda i: i['val_acc'], reverse = True) 
		for i in dict_acc[key]:
			print("\t", i["val_acc"], i["model_struct"])
	
	return

Get_Da_Best()

#==============================================================================
#8. Visualize/Graph

#TESTING OUT ConvLSTM2D - This one isn't working for some reason
"""
lay_gen = Layer_Generator()
model_structures = [["ConvLSTM2D", "ConvLSTM2D", "Flatten"]]
clstm_params = lay_gen.Generate_Simple_Layer_Parameters()["ConvLSTM2D"]

data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 200,
               'o_percent' : 0, #0.25,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)

mt = Model_Tuning(model_structures,
                  dataset,
                  m_tuning = "ConvLSTM2D_Model_Structures",
                  fldr_sffx = '2')
mt.Tune_Models(epochs = 1, batch_size = 3)
    
 

#TESTING OUT THE LOSO CODE
from analysis.zg_Load_Data import Load_Data

data_params = {'dataset' : 'pamap2',
               'train_p' : 0.8,
               'LOSO' : True,
               'w_size' : 400,
               'o_percent' : 0, #0.25,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
"""



