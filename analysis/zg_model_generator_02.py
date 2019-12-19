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
from tensorflow.keras.layers import LeakyReLU
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
                        "o_percent" : [0.0, 0.25, 0.4, 0.5],
                        "w_size" : [0, 10, 25, 50, 100, 200, 400],
                        "train_p" : 0.8
                      }

    return data_parameters

#Generates the possible layer parameters
def Generate_Layer_Parameters():
    layer_parameters = {}
    
    units = [10, 25, 50, 100, 250, 500]
    activation = ["relu", "tanh", "LeakyReLU"]
    bias_init = ["Zeros", "RandomNormal", "glorot_normal"]
    dropout = [0.0, 0.25, 0.5]
    
    layer_parameters["GRU"] = {"units" : units,
                                "activation" : activation,
                                "bias_initializer" : bias_init,
                                "return_sequences" : [True, False],
                                "dropout" : dropout}
    layer_parameters["LSTM"] = {"units" : units,
                                "activation" : activation,
                                "bias_initializer" : bias_init,
                                "return_sequences" : [True, False],
                                "dropout" : dropout}
    layer_parameters["Dense"] = {"units" : units,
                                "activation" : activation,
                                "bias_initializer" : bias_init}
    layer_parameters["BidirectionalLSTM"] = {"layer" : ["LSTM"]}
    layer_parameters["BidirectionalGRU"] = {"layer" : ["GRU"]}

    layer_parameters["Conv1D"] = {"filters" : [0.25, 0.5, 0.75],
                                    "activation" : activation}#, "kernel_size" : filters*kernel_size = window_size?
    print("TODO: Conv1D/ConvLSTM2D filters and kernel_size")
    layer_parameters["ConvLSTM2D"] = {"filters" : [0.25, 0.5, 0.75],
                                    "activation" : activation,
                                    "dropout" : dropout}#, "kernel_size" : filters*kernel_size = window_size?
    print("TODO: filters is currently a percentage of window size, has to be an int at the end")
    #FAUX LAYERS
    layer_parameters["Dropout"] = {"rate" : [0.2, 0.35, 0.5]}
    layer_parameters["MaxPooling1D"] = {"pool_size" : [0.1, 0.2, 0.25]}
    print("TODO: we want strides later... maybe")
    print("TODO: MaxPooling1D - pool_size has to be an integer in the end (based on window size)")
    layer_parameters["Flatten"] = {}
    
    print("TODO: Had to remove print in the Invalid_LayerName area, earch the file for 'TODO'")
    return layer_parameters

def Generate_Layer_Depth():
     return [1, 2, 3, 4, 5, 6, 7]

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
        if d == 1:
            all_models.extend( list(map(lambda el:[el], list_o_layers[0])) )
        else:
            all_models.extend( list(product(*list_o_layers[0:d])) )

    #models_list = all_models
    models_list = Delete_Invalid_Model_Structures(all_models)
    models_list = [list(ele) for ele in models_list] 
    
    with open("data/Model_Structures.pkl", "wb") as fp:   #Pickling
        pickle.dump(models_list, fp)

    return []

#Deletes all of the invalid layer structures
def Delete_Invalid_Model_Structures(all_models):
    itr = 0
    #Loop through the index of each model (in reverse order)
    for index in np.arange(len(all_models))[::-1]:
        #A ton of if statements that are checked
        #If any of the statements are true we delete
        #the whole model from all_models
        
        #Deals with GRU--------------------------------------------------------
        if Invalid_GRU(all_models[index]):
            del all_models[index]
            continue
        #print("1")
        
        #Deals with LSTM-------------------------------------------------------
        if Invalid_LSTM(all_models[index]):
            del all_models[index]
            continue
        #print("2")
        
        #Deals with Dense------------------------------------------------------
        if Invalid_Dense(all_models[index]):
            del all_models[index]
            continue
        #print("3")
        
        #Deals with Bidirectional LSTM-----------------------------------------
        if Invalid_BidirectionalLSTM(all_models[index]):
            del all_models[index]
            continue
        #print("4")
        
        #Deals with Bidirectional GRU------------------------------------------
        if Invalid_BidirectionalGRU(all_models[index]):
            del all_models[index]
            continue
        #print("5")
        
        #Deals with Conv1D-----------------------------------------------------
        if Invalid_Conv1D(all_models[index]):
            del all_models[index]
            continue
        #print("6")
                
        #Deals with ConvLSTM2D-------------------------------------------------
        if Invalid_ConvLSTM2D(all_models[index]):
            del all_models[index]
            continue
        #print("7")
                
        #Deals with Dropout----------------------------------------------------
        if Invalid_Dropout(all_models[index]):
            del all_models[index]
            continue
        #print("8")
                
        #Deals with MaxPooling1D----------------------------------------------------
        if Invalid_MaxPooling1D(all_models[index]):
            del all_models[index]
            continue
        #print("9")
                
        #Flatten---------------------------------------------------------------
        if Invalid_Flatten(all_models[index]):
            del all_models[index]
            continue
        #print("10")
        
        if itr%5000 == 0:
            print(str(itr) + ", ", end="")
        itr = itr + 1
    
    return all_models

def Invalid_GRU(model):
    #print("TODO: Invalid_GRU, check if bidirectional before gru is valid")
    gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
    
    if len(gru_indices) > 0:
        #No dense before gru
        dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
        if len([di for di in dense_indices if di < gru_indices[0]]) > 0:
            return True
        
        #No flatten before GRU
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < gru_indices[0]]) > 0:
            return True

        #Bidir GRU before GRU       Maybe?  TODO: LOOK AT LATER
        #CNN before gru             TRUE
        #ConvLSTM before gru?       TRUE
        #maxpooling1d before gru    TRUE

    return False

def Invalid_LSTM(model):
    #print("TODO: Invalid_LSTM, check if bidirectional before gru is valid")
    lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
    
    if len(lstm_indices) > 0:
        #No dense before lstm
        dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
        if len([di for di in dense_indices if di < lstm_indices[0]]) > 0:
            return True
        
        #No flatten before LSTM
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < lstm_indices[0]]) > 0:
            return True

        #Bidir GRU before lstm       Maybe?  TODO: LOOK AT LATER
        #CNN before lstm             TRUE
        #ConvLSTM before lstm?       TRUE
        #maxpooling1d before lstm    TRUE

    return False

def Invalid_Dense(model):
    return False

def Invalid_BidirectionalLSTM(model):
    blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
    
    if len(blstm_indices) > 0:
        #No dense before blstm
        dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
        if len([di for di in dense_indices if di < blstm_indices[0]]) > 0:
            return True
        
        #No flatten before BLSTM
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < blstm_indices[0]]) > 0:
            return True

        #CNN before it             TRUE
        #ConvLSTM before it?       TRUE
        #maxpooling1d before it    TRUE
        #Bidir before it           TRUE

    return False

def Invalid_BidirectionalGRU(model):
    bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
    
    if len(bgru_indices) > 0:
        #No dense before bgru
        dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
        if len([di for di in dense_indices if di < bgru_indices[0]]) > 0:
            return True
        
        #No flatten before BGRU
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < bgru_indices[0]]) > 0:
            return True

        #CNN before it             TRUE
        #ConvLSTM before it?       TRUE
        #maxpooling1d before it    TRUE
        #Bidir before it           TRUE

    return False

def Invalid_Conv1D(model):
    conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
    
    if len(conv1d_indices) > 0:
        #No RNN style before it
        gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
        if len([fi for fi in gru_indices if fi < conv1d_indices[0]]) > 0:
            return True

        lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
        if len([fi for fi in lstm_indices if fi < conv1d_indices[0]]) > 0:
            return True

        blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
        if len([fi for fi in blstm_indices if fi < conv1d_indices[0]]) > 0:
            return True

        bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
        if len([fi for fi in bgru_indices if fi < conv1d_indices[0]]) > 0:
            return True

        #No flatten before it
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < conv1d_indices[0]]) > 0:
            return True

        #No dense before it
        dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
        if len([di for di in dense_indices if di < conv1d_indices[0]]) > 0:
            return True
        
    return False

def Invalid_ConvLSTM2D(model):
    clstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
    
    if len(clstm_indices) > 0:
        #No RNN style before it
        gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
        if len([fi for fi in gru_indices if fi < clstm_indices[0]]) > 0:
            return True

        lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
        if len([fi for fi in lstm_indices if fi < clstm_indices[0]]) > 0:
            return True

        blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
        if len([fi for fi in blstm_indices if fi < clstm_indices[0]]) > 0:
            return True

        bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
        if len([fi for fi in bgru_indices if fi < clstm_indices[0]]) > 0:
            return True

        #No flatten before it
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < clstm_indices[0]]) > 0:
            return True

        #No dense before it
        dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
        if len([di for di in dense_indices if di < clstm_indices[0]]) > 0:
            return True
        
    return False

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

        #Dropout - faux layer - dropout is INVALID
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        pool_indices = [i for i,d in enumerate(model) if d == 'MaxPooling1D']
        faux_indices = list(flatten_indices + pool_indices)#.sort()
        faux_indices.sort()
        if len(faux_indices) > 0:
            for di in dropout_indices:
                if (di-1) in faux_indices:
                    if (di-2) in dropout_indices:
                        return True
                    if (di-2) in faux_indices:
                        if (di-3) in dropout_indices:
                            return True
                
    #This layer is fine
    return False

def Invalid_MaxPooling1D(model):
    #print("TODO: Invalid_MaxPooling1D, check if it can be the first layer")
    pool_indices = [i for i,d in enumerate(model) if d == 'MaxPooling1D']
    
    if len(pool_indices) > 0:
        #Two pool in a row is bad
        for i in np.arange(len(pool_indices)):
            if i > 0:
                #if any of the indices are next to each other, delete
                if pool_indices[i] == (pool_indices[i-1]+1):
                    return True       
        
        #Can't be after flatten
        flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
        if len([fi for fi in flatten_indices if fi < pool_indices[0]]) > 0:
            return True
        
    return False

def Invalid_Flatten(model):
    
    #Gets the indicies of all the rows that contain dropout
    flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
    
    if len(flatten_indices) > 0:
        #First layer can't be flatten------------------------------------------
        if flatten_indices[0] == 0:
            return True
        #We only want 1 flatten per model structure----------------------------
        if len(flatten_indices) > 1:
            return True

        gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
        lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
        conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
        blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
        bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
        clstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
        
        #Flatten can not be before CNN or RNN type-----------------------------
        #This gets the indices of each CNN/RNN type layer and then checks to see
        #if one of the indices are greater than the location of the flatten index
        if len([i for i in gru_indices if i > flatten_indices[0]]) > 0:
            return True
        if len([i for i in lstm_indices if i > flatten_indices[0]]) > 0:
            return True
        if len([i for i in conv1d_indices if i > flatten_indices[0]]) > 0:
            return True
        if len([i for i in blstm_indices if i > flatten_indices[0]]) > 0:
            return True
        if len([i for i in bgru_indices if i > flatten_indices[0]]) > 0:
            return True
        if len([i for i in clstm_indices if i > flatten_indices[0]]) > 0:
            return True
        
        #Flatten can only be immediately after CNN/ConvLSTM--------------------
        #There can be dropout or max pooling before it also, but the CNN type
        #has to then be before that
        drop_indices = [i for i,d in enumerate(model) if d == 'Dropout']
        pool_indices = [i for i,d in enumerate(model) if d == 'MaxPooling1D']
        flag = True
        #Prev layer is CNN type
        if ((flatten_indices[0]-1) in conv1d_indices) or ((flatten_indices[0]-1) in clstm_indices):
            flag = False
        #2 layers back is CNN type
        if ((flatten_indices[0]-2) in conv1d_indices) or ((flatten_indices[0]-2) in clstm_indices):
            #AND 1 layer back is dropout or max pool
            if ((flatten_indices[0]-1) in drop_indices) or ((flatten_indices[0]-1) in pool_indices):
                flag = False
        #3 layers back is CNN type
        if ((flatten_indices[0]-3) in conv1d_indices) or ((flatten_indices[0]-3) in clstm_indices):
            #AND 2 layers back is dropout or max pool
            if ((flatten_indices[0]-2) in drop_indices) or ((flatten_indices[0]-2) in pool_indices):
                flag = False
                #AND 1 layer back is dropout or max pool
                if ((flatten_indices[0]-1) in drop_indices) or ((flatten_indices[0]-1) in pool_indices):
                    flag = False
        #If none of these cases occur, then flatten is in an invalid location
        if flag:
            return True

        #CNN, Flat               Dense
        #CNN, Drop, Flat         Dense
        #CNN, Flat, Drop         Dense 
        #CNN, Pool, Flat         Dense
        #CNN, Pool, Flat, Drop   Dense
        #CNN, Pool, Drop, Flat   Dense
        #CNN, Drop, Pool, Flat   Dense
    return False

#==============================================================================
#START MODEL CONSTRUCTION AND TUNING

#This function returns the model. For the keras-tuner this is the function
#that it runs
def The_Model(hp): #layer_parameters, model_structures):
    model = Sequential()

    bias_init = None
    model_index = hp.Choice("model_structures_index", model_structures_index)
    chosen_model = model_structures[model_index]

    #Goes through each layer
    for layer_index in np.arange(len(chosen_model)):
        if chosen_model[layer_index] == "GRU":
            bias_init, layers = Add_GRU_Layer(hp, bias_init, chosen_model, layer_index)
            for l in layers:
                model.add(l)
        elif chosen_model[layer_index] == "LSTM":
            bias_init, layers = Add_LSTM_Layer(hp, bias_init, chosen_model, layer_index)
            for l in layers:
                model.add(l)
        elif chosen_model[layer_index] == "Dense":
            bias_init, layers = Add_Dense_Layer(hp, bias_init, chosen_model, layer_index)
            for l in layers:
                model.add(l)
        elif chosen_model[layer_index] == "BidirectionalLSTM":
            bias_init, layers = Add_BidirectionalLSTM_Layer(hp, bias_init, chosen_model, layer_index)
            for l in layers:
                model.add(l)
        elif chosen_model[layer_index] == "BidirectionalGRU":
            bias_init, layer = Add_BidirectionalGRU_Layer(hp, bias_init)
            model.add(layer)
        elif chosen_model[layer_index] == "Conv1D":
            model.add(Add_Conv1D_Layer(hp))
        elif chosen_model[layer_index] == "ConvLSTM2D":
            model.add(Add_ConvLSTM2D_Layer(hp))
        elif chosen_model[layer_index] == "Dropout":
            model.add(Add_Dropout_Layer(hp))
        elif chosen_model[layer_index] == "MaxPooling1D":
            model.add(Add_MaxPooling1D_Layer(hp))
        elif chosen_model[layer_index] == "Flatten":
            model.add(Add_Flatten_Layer(hp))
    
    #TODO: Add in final layer
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #print(model.get_config())
    #print(model.summary())
    print("The_Model is not completed")
    return model

def Add_GRU_Layer(hp, bias_init, all_layers, layer_index):
    #Because of leakyReLU, we technically return multiple layers
    these_layers = []
    
    #The importance of this depends on how keras-tuner stores the parameters,
    #im assuming that it stores them by like a dictionary.
    name_prefix = "GRU_" + str(layer_index) + "_"
    
    #We do this here just because we can
    layer_parameters = Generate_Layer_Parameters()["GRU"]
    
    #Random choice for these two parameters
    units = hp.Choice(name_prefix + "units", layer_parameters["units"])
    dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])

    #all bias_initializer in the model should be the same
    if bias_init == None:
        print("Set the bias_initializer from the random thingy")
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
    else:
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
    
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    if bias_initializer == "Zeros":
        activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
    elif bias_initializer == "glorot_normal":
        activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
    elif bias_initializer == "RandomNormal":
        activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])

    #if a layer after this is a RNN type or a Dropout-RNN type then
    #return_sequences can be either true or false, otherwise it is false
    if len(all_layers) > (layer_index + 1):
        rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
        if all_layers[layer_index + 1] in rnn_types:    #GRU-RNN type
            return_sequences = hp.Choice(name_prefix + "return_sequences_t", [True]) # layer_parameters["return_sequences"])
        else:
            if all_layers[layer_index + 1] == "Dropout":
                if len(all_layers) > (layer_index + 2):
                    if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
                        return_sequences = hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
                    else:
                        return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
                else:
                    return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
            else:
                return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
    else:
        return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])

    #If this is the first layer in the model, this has to have the input shape
    #fed into it
    if layer_index == 0:
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = GRU(units = units, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2]))
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = GRU(units = units, activation = activation, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2]))
            these_layers.append(layer)
    else: 
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = GRU(units = units, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer)
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = GRU(units = units, activation = activation, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer)
            these_layers.append(layer)
            
    return (bias_init, these_layers)

def Add_LSTM_Layer(hp, bias_init, all_layers, layer_index):
    #Because of leakyReLU, we technically return multiple layers
    these_layers = []
    
    #The importance of this depends on how keras-tuner stores the parameters,
    #im assuming that it stores them by like a dictionary.
    name_prefix = "LSTM_" + str(layer_index) + "_"
    
    #We do this here just because we can
    layer_parameters = Generate_Layer_Parameters()["LSTM"]
    
    #Random choice for these two parameters
    units = hp.Choice(name_prefix + "units", layer_parameters["units"])
    dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])

    #all bias_initializer in the model should be the same
    if bias_init == None:
        print("Set the bias_initializer from the random thingy")
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
    else:
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
    
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    if bias_initializer == "Zeros":
        activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
    elif bias_initializer == "glorot_normal":
        activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
    elif bias_initializer == "RandomNormal":
        activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])

    #if a layer after this is a RNN type or a Dropout-RNN type then
    #return_sequences can be either true or false, otherwise it is false
    if len(all_layers) > (layer_index + 1):
        rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
        if all_layers[layer_index + 1] in rnn_types:    #GRU-RNN type
            return_sequences = hp.Choice(name_prefix + "return_sequences_t", [True])# layer_parameters["return_sequences"])
        else:
            if all_layers[layer_index + 1] == "Dropout":
                if len(all_layers) > (layer_index + 2):
                    if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
                        return_sequences = hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
                    else:
                        return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
                else:
                    return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
            else:
                return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
    else:
        return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])
    
    #If this is the first layer in the model, this has to have the input shape
    #fed into it
    if layer_index == 0:
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = LSTM(units = units, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2]))
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = LSTM(units = units, activation = activation, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2]))
            these_layers.append(layer)
    else: 
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = LSTM(units = units, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer)
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = LSTM(units = units, activation = activation, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer)
            these_layers.append(layer)
            
    return (bias_init, these_layers)

#Adds the Dense layer and hyperparameters by condition
def Add_Dense_Layer(hp, bias_init, all_layers, layer_index):
    #Because of leakyReLU, we technically return multiple layers
    these_layers = []
    
    #The importance of this depends on how keras-tuner stores the parameters,
    #im assuming that it stores them by like a dictionary.
    name_prefix = "Dense_" + str(layer_index) + "_"
    
    #We do this here just because we can
    layer_parameters = Generate_Layer_Parameters()["Dense"]
    
    #Random choice for these two parameters
    units = hp.Choice(name_prefix + "units", layer_parameters["units"])
    #dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])

    #all bias_initializer in the model should be the same
    if bias_init == None:
        print("Set the bias_initializer from the random thingy")
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
    else:
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
    
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    if bias_initializer == "Zeros":
        activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
    elif bias_initializer == "glorot_normal":
        activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
    elif bias_initializer == "RandomNormal":
        activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])

    #If this is the first layer in the model, this has to have the input shape
    #fed into it
    if layer_index == 0:
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = Dense(units = units,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2]))
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = Dense(units = units, activation = activation,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2]))
            these_layers.append(layer)
    else: 
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = Dense(units = units,
                        bias_initializer = bias_initializer)
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = Dense(units = units, activation = activation,
                        bias_initializer = bias_initializer)
            these_layers.append(layer)
            
    return (bias_init, these_layers)

def Add_BidirectionalLSTM_Layer(hp, bias_init, all_layers, layer_index):
    #Because of leakyReLU, we technically return multiple layers
    these_layers = []
    
    #The importance of this depends on how keras-tuner stores the parameters,
    #im assuming that it stores them by like a dictionary.
    name_prefix = "BidirectionalLSTM_" + str(layer_index) + "_"
    
    #We do this here just because we can
    layer_parameters = Generate_Layer_Parameters()["LSTM"]
    
    #Random choice for these two parameters
    units = hp.Choice(name_prefix + "units", layer_parameters["units"])
    dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])

    #all bias_initializer in the model should be the same
    if bias_init == None:
        print("Set the bias_initializer from the random thingy")
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
    else:
        bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
    
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    if bias_initializer == "Zeros":
        activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
    elif bias_initializer == "glorot_normal":
        activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
    elif bias_initializer == "RandomNormal":
        activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])

    #if a layer after this is a RNN type or a Dropout-RNN type then
    #return_sequences can be either true or false, otherwise it is false
    if len(all_layers) > (layer_index + 1):
        rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
        if all_layers[layer_index + 1] in rnn_types:    #GRU-RNN type
            return_sequences = hp.Choice(name_prefix + "return_sequences_t", [True])# layer_parameters["return_sequences"])
        else:
            if all_layers[layer_index + 1] == "Dropout":
                if len(all_layers) > (layer_index + 2):
                    if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
                        return_sequences = hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
                    else:
                        return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
                else:
                    return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
            else:
                return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])
    else:
        return_sequences = hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])
    
    #If this is the first layer in the model, this has to have the input shape
    #fed into it
    if layer_index == 0:
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = Bidirectional(LSTM(units = units, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2])))
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = Bidirectional(LSTM(units = units, activation = activation, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer,
                        input_shape = (dataset.x_train.shape[1], dataset.x_train.shape[2])))
            these_layers.append(layer)
    else: 
        #leakyReLU is a pain because it acts like it is its own layer
        if activation == "LeakyReLU":
            layer = Bidirectional(LSTM(units = units, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer))
            these_layers.append(layer)
            these_layers.append(LeakyReLU())
        else:
            layer = Bidirectional(LSTM(units = units, activation = activation, dropout = dropout,
                        return_sequences = return_sequences,
                        bias_initializer = bias_initializer))
            these_layers.append(layer)
            
    return (bias_init, these_layers)

def Add_BidirectionalGRU_Layer(hp, bias_init):
    #all bias_initializer in the model should be the same

    #activation function can be different across layers
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    
    #layer after return_sequences has to be a RNN style layer

    print("Add_BidirectionalGRU_Layer not complete and return actual hyperparameters")
    return (bias_init, Bidirectional(GRU(100)))

def Add_Conv1D_Layer():
    #Convert filter percent to a int by multiplying it times the window_size
    #kernel_size is window_size/filter (for our purposes)

    #activation function can be different across layers
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    
    print("Add_Conv1D_Layer not complete and return actual hyperparameters")
    return Conv1D(100, 40)

def Add_ConvLSTM2D_Layer():
    #Convert filter percent to a int by multiplying it times the window_size
    #kernel_size is window_size/filter (for our purposes)

    #activation function can be different across layers
    #when activation is relu, use RandomNormal or Zero
    #when activation is leakyRelu, can use RandomNormal or Zero
    #when activation is tanh, use glorot_normal (Xavier) or Zero
    
    print("Add_ConvLSTM2D_Layer not complete and return actual hyperparameters")
    return Conv1D(100, 40)

#Adds the Dropout layer and hyperparameters by condition
def Add_Dropout_Layer():
    print("Add_Dropout_Layer return actual hyperparameters")
    return Dropout(0.5)

def Add_MaxPooling1D_Layer():
    #pool_size is currently a percent, multiply it by window_size to get an int
    
    print("Add_MaxPooling1D_Layer not complete and return actual hyperparameters")
    return MaxPooling1D(0.5)

def Add_Flatten_Layer():
    return Flatten()

#==============================================================================
#START RUNNING THE MODELS

"""
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

#Tunes the model with the given data parameters
def Tune_Models(dataset):
    MAX_TRIALS = 5
    EXECUTIONS_PER_TRIAL = 5
    
    tuner = RandomSearch(
            The_Model,
            objective = 'val_accuracy',
            max_trials = MAX_TRIALS,
            executions_per_trial = EXECUTIONS_PER_TRIAL,
            directory = 'data\\test_dir\\',
            project_name = 'tune_optimizer2\\',
            seed = 42
        )
    
    tuner.search(x = dataset.x_train,
                 y = dataset.y_train,
                 epochs = 3,
                 batch_size = 4,
                 validation_data = (dataset.x_test, dataset.y_test))
    
    models = tuner.get_best_models(num_models=2)
    
    #Model 0's setup
    print(models[0].get_config())
    
    return []

#==============================================================================
#==============================================================================

#Generating the model structures can be done separately so its here
Generate_Model_Strutures(Generate_Layer_Parameters(), Generate_Layer_Depth())

#keras-tuner can not take in an array, it has to be string, int, float
#so we have it select the index of the structure, instead of the actual model
#model_structures = [["GRU"], ["GRU", "GRU"]]# Load_Model_Structures()
model_structures = [["BidirectionalLSTM"], ["BidirectionalLSTM", "BidirectionalLSTM"]]
model_structures_index = [0, 1]

#data_parameters = Generate_Data_Parameters()
data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 200,
               'o_percent' : 0.25
               }
dataset = Load_Data(**data_params)

#Does the hyperparameter tuning of the models on the given dataset
Tune_Models(dataset)


