from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from pandas import read_csv
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
from tensorflow.keras.utils import to_categorical

#import os
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data

#import warnings
#warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#START| RUNNING MODEL WITH ACCURACY

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(x_train, y_train, x_test, y_test, model, model_params, repeats = 5):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = model(x_train, y_train, x_test, y_test, **model_params)
		#score = evaluate_model_lstm(x_train, y_train, x_test, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

#------------------------------------------------------------------------------
#START| MODEL STUFFS
    
#Fit and evaluate a model - 33.3%
def model_cnn_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 8):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(Dropout(0.2))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#Fit and evaluate a model - 100% accuracy 11.10.19
def model_gru_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	model = Sequential()
	model.add(GRU(500, input_shape=(n_timesteps, n_features), return_sequences = False))
	model.add(Dropout(0.25))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#Fit and evaluate a model - 100% accuracy 11.10.19
def model_lstm_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	model = Sequential()
	model.add(LSTM(500, input_shape=(n_timesteps, n_features), return_sequences = False))
	model.add(Dropout(0.25))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#Fit and evaluate a model - 100% accuracy 11.10.19
def model_bidirlstm_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	model = Sequential()
	model.add(Bidirectional(LSTM(500, input_shape=(n_timesteps, n_features), return_sequences = False)))
	model.add(Dropout(0.25))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#for PAMAP2 dataset - 94.99% accuracy 11.10.19
#Fit and evaluate a model - 99.89% accuracy 11.10.19
def model_nn_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 100):
	model = Sequential()
	model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

def model_convlstm_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 8):
	# define model
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	# reshape into subsequences (samples, time steps, rows, cols, channels)
	n_steps = 4
	n_length = int(x_train.shape[1]/n_steps)
	x_train = x_train.reshape((x_train.shape[0], n_steps, 1, n_length, n_features))
	x_test = x_test.reshape((x_test.shape[0], n_steps, 1, n_length, n_features))

	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#------------------------------------------------------------------------------
#START| MAIN

#Load the data with the fixed parameters into memory
data_params = {'dataset' : 'firebusters',
	               'train_p' : 0.8,
                   'w_size' : 200,
                   'o_percent' : 0.25,
	   		       'LOSO' : True,
                   'clstm_params' : {}
                   }
dataset = Load_Data(**data_params)

#dataset, train_p = 0.8, w_size = 0, o_percent = 0.25):
#data_params = {'dataset' : 'firebusters',
#               'train_p' : 0.8,
#               'w_size' : 400,
#               'o_percent' : 0.25
#               }
#dataset = Load_Data(**data_params)

model_params = {'epochs' : 10,
                'batch_size' : 32
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_01, model_params, 5)



