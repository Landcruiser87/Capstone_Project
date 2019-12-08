from numpy import mean
from numpy import std
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from kerastuner.tuners import RandomSearch
from itertools import permutations
from itertools import product
import numpy as np

import os
os.chdir("C:/githubrepo/TMP/") #TEST
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data

import warnings
warnings.filterwarnings("ignore")

def Get_Data_Parameters():
    data_parameters = {
                        "dataset" : ["har", "pamap2", "firebusters"],
                        "overlap_percent" : [0, 0.25, 0.4, 0.5],
                        "window_size" : [10, 25, 50, 100, 200, 400]
                      }

    return data_parameters

def Get_Layer_Parameters():
    layer_parameters = {}
    
    layer_parameters["Dense"] = {"units" : [10, 25, 50, 100, 250, 500],
                                 "activation" : ["relu", "tanh", "linear"]}
    layer_parameters["GRU"] = {"units" : [10, 25, 50, 100, 250, 500],
                                "activation" : ["relu", "tanh", "linear"],
                                "dropout" : [0, 0.25, 0.5]}
    layer_parameters["Dropout"] = {"rate" : [0.2, 0.4, 0.5]}
    
    return layer_parameters

def Generate_Model_Strutures(layer_p, depth):
    all_models = []
    
    #Get all the layer names
    layer_names = [key for key in layer_p]
    
    #This will be used for making all the permutations of layers
    list_o_layers = []
    for _ in np.arange(max(depth)):
        list_o_layers.append(layer_names)
    
    #Makes all the layer permutations for each desired depth
    for d in depth:
        all_models.extend( list(product(*list_o_layers[0:d])) )
    
    #TODO: Delete all invalid layer orders
    return Delete_Invalid_Model_Structures(all_models)

def Invalid_Dropout(model):
    #Gets the indicies of all the rows that contain dropout
    dropout_indices = [i for i,d in enumerate(model) if d == 'Dropout']

    if len(dropout_indices) > 0:
        #Makes sure the first layer isn't dropout
        if dropout_indices[0] == 0:
            return True
        #Eliminates dropout two in a row
        for i in np.arange(len(dropout_indices)):
            if i > 0:
                #if any of the indices are next to each other, delete
                if dropout_indices[i] == (dropout_indices[i-1]+1):
                    return True
    
    #This layer is fine
    return False

def Delete_Invalid_Model_Structures(all_models):
    
    #Loop through the index of each model (in reverse order)
    for index in np.arange(len(all_models))[::-1]:
        #A ton of if statements that it checks
        #If any of the statements are true we delete
        #the whole model from all_models
        
        #Deals with Dropout----------------------------------------------------
        if Invalid_Dropout(all_models[index]):
            del all_models[index]
            continue
                
        #Deals with GRU--------------------------------------------------------
        
        #Deals with Dense------------------------------------------------------

        #Deals with etc...-----------------------------------------------------
    
    print("Delete_Invalid_Model_Structures function incomplete")
    return all_models

def Model_Generator():
    data_parameters = Get_Data_Parameters()
    layer_parameters = Get_Layer_Parameters()
    layer_depth = [2, 3]#, 4]#, 5, 6, 7]
    
    model_structures = Generate_Model_Strutures(layer_parameters,
                                                layer_depth)
    
    print(model_structures)
    return []


Model_Generator()

