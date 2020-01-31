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
					  parent_fldr = "step2",
                      fldr_sffx = '1')
	#Runs the model tuner, in this instance it is Layer Tuning
    mt.Tune_Models(epochs = 500, batch_size = 64, MAX_TRIALS = 1000)
    
    return

#A list of all the categories that could be run
#The layers to run on this iteration (not doing all at the same time because its slow)
#Running the layer tuner
"""["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D", "Dense", "GRU", "LSTM"]"""
categories = ["ConvLSTM2D"]
Find_All_Layers_Accuracy(categories)
