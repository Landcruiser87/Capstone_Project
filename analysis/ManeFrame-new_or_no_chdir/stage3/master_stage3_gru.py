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
#import os
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator
from analysis.zg_model_tuning_01 import Model_Tuning
from analysis.ah_Model_Info import Model_Info

#import warnings
#warnings.filterwarnings("ignore")

#==============================================================================
#3. Hyperparameter Tuning

def Run_Hyper_On_Best_By_Category(categories):
	mi = Model_Info()
	parent_folder = "step2"
	best_by_type = mi.Get_Best_Layer_Structure_Types(best_x = 15, parent_folder = parent_folder)
	
	for key in best_by_type:
		if key in categories:
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
					  parent_fldr = "step3",   #'Project' folder name
					  fldr_name = layer_type, #This tuning's folder name
                      fldr_sffx = '1')        #Suffix for the folder just in case
	mt.Tune_Models(epochs = 60, batch_size = 64)
    
	return

"""["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D", "Dense", "GRU", "LSTM"]"""
categories = ["GRU"]
Run_Hyper_On_Best_By_Category(categories)
