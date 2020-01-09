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

#Makes an instance of the class
lay_gen = Layer_Generator()

#Generates and saves to disk the huge list of all model layer setups
lay_gen.Generate_Model_Strutures(lay_gen.Generate_Layer_Parameters(), lay_gen.Generate_Layer_Depth())

#Saves and splits up that huge list of all saved layer setups into categories
#saves into the data folder
lay_gen.Split_Save_Models_Into_Categories()

#==============================================================================
#2. Figure out what the parameters for each category

#NOT REALLY A STEP TO RUN, JUST HERE FOR LOOKING AT... OOO, AHHHH
params = lay_gen.Generate_Simple_Layer_Parameters()

#==============================================================================
#3. Run each category with specific hyperparameters

#To find what the best layer setups are
def Find_All_Layers_Accuracy():
#    names = ["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D",
#             "Dense", "GRU", "LSTM", "Other"]
    names = ["GRU", "LSTM"]
    
    for layer_type in names:
        Find_Layer_Accuracy(layer_type)
    
    return

def Find_Layer_Accuracy(layer_type):
    layer_type = layer_type + "_Model_Structures"
    
    #Run tuning on this with the given name
    lay_gen = Layer_Generator()
    model_structures = lay_gen.Load_Model_Structures(layer_type)
    
    clstm_params = []
    if layer_type == "ConvLSTM2D_Model_Structures":
        clstm_params = lay_gen.Generate_Simple_Layer_Parameters()["ConvLSTM2D"]
    
    data_params = {'dataset' : 'firebusters',
                   'train_p' : 0.8,
                   'w_size' : 400,
                   'o_percent' : 0, #0.25,
                   'clstm_params' : clstm_params
                   }
    dataset = Load_Data(**data_params)
    
    mt = Model_Tuning(model_structures,
                      dataset,
                      m_tuning = layer_type,
                      fldr_sffx = '1')
    mt.Tune_Models(epochs = 1, batch_size = 300)
    
    return

#Run it!
Find_All_Layers_Accuracy()

#==============================================================================
#4. Take BEST 15 for each category (15LSTM, 15GRU...)
#5. Run hyperparameter tuning for BEST of each category


#6. Pull out TOP 10 for each tuned
#7. Visualize/Graph


#TESTING OUT ConvLSTM2D - This one isn't working for some reason
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
    
 
print(lstm_parmas)   
    
    