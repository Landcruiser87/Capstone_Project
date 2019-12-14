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
import pickle

import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#START SETTING UP THE PARAMETERS/LAYERS FOR THE MODELS
#Generates the data parameters
def Generate_Data_Parameters():
    data_parameters = {
                        "dataset" : ["har", "pamap2", "firebusters"],
                        "o_percent" : [0, 0.25, 0.4, 0.5],
                        "w_size" : [0, 10, 25, 50, 100, 200, 400],
                        "train_p" : 0.8
                      }

    return data_parameters

#Generates the possible layer parameters
def Generate_Layer_Parameters():
    layer_parameters = {}
    
    layer_parameters["Dense"] = {"units" : [10, 25, 50, 100, 250, 500],
                                 "activation" : ["relu", "tanh", "linear"]}
    layer_parameters["GRU"] = {"units" : [10, 25, 50, 100, 250, 500],
                                "activation" : ["relu", "tanh", "linear"],
                                "dropout" : [0, 0.25, 0.5]}
    layer_parameters["Dropout"] = {"rate" : [0.2, 0.4, 0.5]}
    
    return layer_parameters

def Generate_Layer_Depth():
     return [2, 3]#, 4]#, 5, 6, 7]

#If you have created the model structure already, this loads it in from a file
def Load_Model_Structures():
    model_structures = []
    
    # open file and read the content in a list
    with open("data/Model_Structures.pkl", "rb") as fp:   # Unpickling
        model_structures = pickle.load(fp)
    
    return model_structures

#Generates all of the structures possible for the layers in a model then
#saves it to a file
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
    
    models_list = Delete_Invalid_Model_Structures(all_models)
    
    with open("data/Model_Structures.pkl", "wb") as fp:   #Pickling
        pickle.dump(models_list, fp)

    return []

#Deletes all of the invalid layer structures
def Delete_Invalid_Model_Structures(all_models):
    
    #Loop through the index of each model (in reverse order)
    for index in np.arange(len(all_models))[::-1]:
        #A ton of if statements that are checked
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

#If the way Dropout is set in the model is invalid, it markes it for removal
def Invalid_Dropout(model):
    #Gets the indicies of all the rows that contain dropout
    dropout_indices = [i for i,d in enumerate(model) if d == 'Dropout']

    if len(dropout_indices) > 0:
        #First layer can't be dropout
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

#==============================================================================
#START MODEL CONSTRUCTION AND TUNING

#Adds the Dropout layer and hyperparameters by condition
def Add_Dropout_Layer():
    print("Add_Dropout_Layer not completed")
    return Dropout(0.5)

#Adds the Dense layer and hyperparameters by condition
def Add_Dense_Layer(layer_index):
    print("Add_Dense_Layer not completed")
    
    if layer_index == 0:
        Dense(200, input_dim = 57, activation = 'relu')

    return Dense(200, activation='relu')

#This function returns the model. For the keras-tuner this is the function
#that it runs
def The_Model(layer_parameters, model_structures):
    model = Sequential()

    #Goes through each layer
    for layer_index in np.arange(len(model_structures)):
        if model_structures[layer_index] == "Dense":
            continue
        elif model_structures[layer_index] == "GRU":
            continue
        elif model_structures[layer_index] == "Dropout":
            return model.add(Add_Dropout_Layer(model_structures, layer_parameters, layer_index))
    
    #TODO: Add in final layer
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("The_Model is not completed")
    return model

#==============================================================================
#START RUNNING THE MODELS

#Tunes the model with the given data parameters
def Tune_Models(dataset, data_parameters):
    layer_parameters = Generate_Layer_Parameters()
    layer_depth = Generate_Layer_Depth()
    model_structures = Load_Model_Structures()

    #TODO: Select the data parameters to use
    print("TODO: Select the data parameters")
    
    model = The_Model(layer_parameters, model_structures[0])
    model.fit(dataset.x_train, dataset.y_train,
              validation_data=(dataset.x_test, dataset.y_test),
              epochs = 3, batch_size = 300)
    
    """
    MAX_TRIALS = 20
    EXECUTIONS_PER_TRIAL = 5
    tuner = RandomSearch(
            The_Model,
            objective = 'val_accuracy',
            max_trials = MAX_TRIALS,
            executions_per_trial = EXECUTIONS_PER_TRIAL,
            directory = 'data/test_dir',
            project_name = 'tune_optimizer',
            seed = 42
        )
    """
    return []

#==============================================================================
#==============================================================================

#Generating the model structures can be done separately so its here
Generate_Model_Strutures(Generate_Layer_Parameters(), Generate_Layer_Depth())

#data_parameters = Generate_Data_Parameters()
data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 0,
               'o_percent' : 0.25
               }
dataset = Load_Data(**data_params)

#Does the hyperparameter tuning of the models on the given dataset
Tune_Models(dataset, data_params)


