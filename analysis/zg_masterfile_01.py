"""
        1. Split models into categories
        	All Separate
        	Combo
        2. Figure out what the parameters for each category
        	Units = 10
        	activation = relu
        	bias_init = zeros
        	dropout = 0.25
        	Conv1D
        		filters = 0.75
        	ConvLSTM2D
        		filters = 0.75
        	Dropout
        		rate = 0.2
        	MaxPooling1D
        		pool_size = 0.25
        3. Run each category with specific hyperparameters
        4. Take BEST 15 for each category (15LSTM, 15GRU...)
        5. Run hyperparameter tuning for BEST of each category
        6. Pull out TOP 10 for each tuned
        7. Visualize/Graph
"""

import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator
from analysis.zg_model_tuning_01 import Model_Tuning

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
# 1. Split models into categories

#Makes an instance of the class.. not sure it necessary
lay_gen = Layer_Generator()

#Generates and saves to disk the huge list of all model layer setups
lay_gen.Generate_Model_Strutures(lay_gen.Generate_Layer_Parameters(), lay_gen.Generate_Layer_Depth())

#Saves and splits up that huge list of all saved layer setups into categories
lay_gen.Split_Save_Models_Into_Categories()

#==============================================================================
#2. Figure out what the parameters for each category

params = lay_gen.Generate_Simple_Layer_Parameters()

#==============================================================================
#3. Run each category with specific hyperparameters

#To find what the best layer setups are
def Find_Layers_Accuracy():
    suffix = "_Model_structures"
    names = ["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D",
             "Dense", "GRU", "LSTM", "Other"]
    
    for layer_type in names:
        Find_Layer_Accuracy(layer_type)
    
    return

def Find_Layer_Accuracy(layer_type):
    
    #Run tuning on this with the given name
    
    return

#Run it!
#Find_Layers_Accuracy()


gen = Layer_Generator()
GRU_model_structures = gen.Load_Model_Structures("GRU_Model_Structures")

data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 200,
               'o_percent' : 0 #0.25
               }
dataset = Load_Data(**data_params)

mt = Model_Tuning(GRU_model_structures, dataset, m_tuning = "GRU_Model_Structures")
mt.Tune_Models(epochs = 1, batch_size = 300)


print(mt.model_structures_index)
print(type(mt.model_structures_index))
print(len(mt.model_structures))

print(len(mt.model_structures_index))
print(type(mt.model_structures_index))



#==============================================================================


