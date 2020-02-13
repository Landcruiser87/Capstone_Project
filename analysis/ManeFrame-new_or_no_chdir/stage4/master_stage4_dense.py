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
import random
import os
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator
from analysis.zg_model_tuning_01 import Model_Tuning
from analysis.ah_Model_Info import Model_Info

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#4. Data Tuning

def Run_Data_Hyperparameter_Tuning(categories):
	#The way that this program gets the model_struct is by looking at the
	#model_index. Because we are not using the full model index and just the
	#best ones from the previous run, we have to get them and use their index
	#to get the real model structure.

	mi = Model_Info()
	bbt_s2 = mi.Get_Best_Layer_Structure_Types(best_x = 15, parent_folder = "step2")
	bbt_s3 = mi.Get_Best_Layer_Structure_Types_With_Hyperparameters(best_x = 10, parent_folder = "step3")
	
	#Looping by type
	for key in bbt_s3:
		#Looping over the individual results
		for result in bbt_s3[key]:
			#Gets the actual model structures index from the previous best
			#and overrides the invalid ones
			result['model_struct'] = bbt_s2[key][result['model_index']]
	
	s = mi.Restructure_Layer_Hyp(bbt_s3)
	
	#Loop over the types in the best by type for stage 3
	for key in bbt_s3:
		#For running the program we check if we are running this type of category
		if key in categories:
			#Need to pull out the layers:
			model_structures = []
			for item in s[key]:
				model_structures.append(item[0])
			
			Data_Hyperparameter_Tuning(key, model_structures, s[key])

	return
	
def Data_Hyperparameter_Tuning(model_structures_type, model_structures, hyp_str):
	window_size = [0]
	overlap_percent = [0]
	batch_size = [16, 64, 256]

	#Makes all combinations of the three parameters's values
	all_data_parameters = [window_size, overlap_percent, batch_size]
	all_data_parameters = list(itertools.product(*all_data_parameters))
	
	#Shuffle the parameters so random combos get chosen
	random.seed(1)
	random.shuffle(all_data_parameters)
	
	#Used for creating the folders in the tuner
	loop_num = 0
	
	#Save the parameters to the data folder
	with open("data/step4_hyp/" + model_structures_type + "_ModelStr_Hyp.pkl", "wb") as fp:   #Pickling
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
	                      m_tuning = "data_" + model_structures_type,	#Whether to use simple or all hyperparamters
						  parent_fldr = "step4",   #'Project' folder name
						  fldr_name = layer_type + "_" + model_structures_type + "_", #This tuning's folder name
	                      fldr_sffx = str(loop_num))        #Suffix for the folder
		mt.Tune_Models(epochs = 60, batch_size = params[2], MAX_TRIALS = 20)

		#Save the data parameters to the file
		curFolder = layer_type + "_" + model_structures_type + "_" + str(loop_num)
		with open("data\\step4\\/" + curFolder + "\\/data_parameters.pkl", "wb") as fp:   #Pickling
			pickle.dump([params, model_structures], fp)

		loop_num += 1

	#Delete the pkl file
	os.remove("data/step4_hyp/" + model_structures_type + "_ModelStr_Hyp.pkl") 

	return

"""["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D", "Dense", "GRU", "LSTM"]"""
categories = ["Dense"]
Run_Data_Hyperparameter_Tuning(categories)


