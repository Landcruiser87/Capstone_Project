from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
#from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
import analysis.zg_Load_Data

import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
#os.chdir("C:/SAM SAM SAM SAM/CapstoneA/data/") #Sam's github data folder
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#START| MODEL STUFFS

#Fit and evaluate a model - 33.3%
def evaluate_model_cnn(x_train, y_train, x_test, y_test):
	epochs, batch_size = 10, 8
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
	model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

	# evaluate model
	_, accuracy = model.evaluate(x_test, y_test, batch_size = batch_size)

	return accuracy

#Fit and evaluate a model - 100% accuracy 11.10.19
def evaluate_model_lstm(x_train, y_train, x_test, y_test):
	epochs, batch_size = 10, 4
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
	model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

	# evaluate model
	_, accuracy = model.evaluate(x_test, y_test, batch_size = batch_size)

	return accuracy

#for PAMAP2 dataset - 94.99% accuracy 11.10.19
#Fit and evaluate a model - 99.89% accuracy 11.10.19
def evaluate_model(x_train, y_train, x_test, y_test):
	epochs, batch_size = 10, 4
	
	model = Sequential()
	#model.add(Embedding(57, 32, input_length = 57))
	#model.add(LSTM(32))
	model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

	# evaluate model
	_, accuracy = model.evaluate(x_test, y_test, batch_size = batch_size)

	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(x_train, y_train, x_test, y_test, repeats = 5):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model_cnn(x_train, y_train, x_test, y_test)
		#score = evaluate_model_lstm(x_train, y_train, x_test, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

#------------------------------------------------------------------------------
#START| MAIN

#dataset, train_p = 0.8, w_size = 0, o_percent = 0.25):
data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 0,
               'o_percent' : 0.25
               }
dataset = Load_Data(**data_params)
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, 3)



	