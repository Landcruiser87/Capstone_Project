import numpy as np
import pandas as pd
import os
os.chdir('/Users/samanthasprague/Capstone/Capstone_Project')
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator
from analysis.zg_model_tuning_01 import Model_Tuning
from analysis.ah_Model_Info import Model_Info

def SAM(layer_type):
    layer_type = layer_type + "_Model_Structures"
    model_structures = [["GRU"],["GRU", "GRU"]]
    data_params = {'dataset' : 'firebusters',
                   'train_p' : 0.8,
                   'w_size' : 400,
                   'o_percent' : 0, #0.25,
                   'clstm_params' : {}
                   }
    dataset = Load_Data(**data_params)
    mt = Model_Tuning(model_structures,
                      dataset,
                      m_tuning = layer_type,
                                          fldr_name = layer_type,
                                          parent_fldr = "step3",
                      fldr_sffx = '1')
    mt.Tune_Models(epochs = 3, batch_size = 300)
    return
SAM("SAM")
