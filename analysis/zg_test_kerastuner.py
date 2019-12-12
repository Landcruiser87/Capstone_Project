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
from IPython.display import display, HTML

import os
os.chdir("C:/githubrepo/CapstoneA/") #TEST
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data

import warnings
warnings.filterwarnings("ignore")

def tune_optimizer_model(hp):
    model = Sequential()
    model.add(Dense
        (
            units = 40,
            activation = "relu",
            input_shape = [dataset.x_train.shape[1]]
        ))
	#model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))

    model.add(Dense(dataset.y_train.shape[1], activation='softmax'))

    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])

    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    return model

#dataset, train_p = 0.8, w_size = 0, o_percent = 0.25):
data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 0,
               'o_percent' : 0.25
               }
dataset = Load_Data(**data_params)

MAX_TRIALS = 20
EXECUTIONS_PER_TRIAL = 5
tuner = RandomSearch(
        tune_optimizer_model,
        objective = 'val_accuracy',
        max_trials = MAX_TRIALS,
        executions_per_trial = EXECUTIONS_PER_TRIAL,
        directory = 'data/test_dir',
        project_name = 'tune_optimizer',
        seed = 42
    )

#Only works in a ipynb (HTML Stuffs)
tuner.search_space_summary()

#Will load the previously run stuff from file (i think)
tuner.reload()

#Unfortunately prints out info that is only readable in a ipynb (HTML Stuffs)
tuner.search(x=dataset.x_train,
             y=dataset.y_train,
             epochs=3,
             validation_data=(dataset.x_test, dataset.y_test))

#Shows the summary of the top results? (only works in ipynb)
tuner.results_summary()

#Gets the top 2 models
models = tuner.get_best_models(num_models=2)
#Model 0's setup
models[0].get_config()


