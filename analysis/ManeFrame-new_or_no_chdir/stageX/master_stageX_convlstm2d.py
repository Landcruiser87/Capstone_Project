from numpy import mean
from numpy import std
import numpy as np
import pandas as pd
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator

import warnings
warnings.filterwarnings("ignore")

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
#EXAMPLE MODEL STUFFS
'''
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
'''
#==============================================================================
#DONE ONES
#------------------------------------------------------------------------------
#Conv1D
''' #5
[[['Conv1D', 'Conv1D', 'Conv1D', 'MaxPooling1D', 'Conv1D', 'Flatten'], [1.0], [28.248740702499582], [1.0], [23.140694618225098], [200, 25, 64]], {'Conv1D_0': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Conv1D_1': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Conv1D_2': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'MaxPooling1D_3': {'pool_size': [0.2]}, 'Conv1D_4': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'optimizer': 'RMSprop'}]
'''
def model_cnn_05(x_train, y_train, x_test, y_test, epochs = 60, batch_size = 64, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	f_size = max(1, int(window_size*0.25))
	p_size = max(1, int(0.2*window_size))
	k_size = max(2, int(f_size*0.1))

	model = Sequential()
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', input_shape=(n_timesteps,n_features), padding="same"))
	model.add(LeakyReLU())
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(MaxPooling1D(pool_size=p_size))
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Flatten())
	model.add(Dense(n_outputs, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	# fit network
	#es = [EarlyStopping(monitor='val_accuracy', mode='max', patience = 8, restore_best_weights=True)]
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)#, callbacks=es)

	return results.history["val_accuracy"][-1]

'''#16
[[['Conv1D', 'Dropout', 'Conv1D', 'Conv1D', 'Conv1D', 'Dropout', 'Flatten'], [0.9734151363372803], [111.47861522448332], [1.0], [281.3941955566406],  [400, 0, 64]], {'Conv1D_0': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_1': {'rate': [0.35]}, 'Conv1D_2': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Conv1D_3': {'activation': ['LeakyReLU'], 'filters': [0.25]},'Conv1D_4': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_5': {'rate': [0.5]}, 'optimizer': 'adam'}]
'''
def model_cnn_16(x_train, y_train, x_test, y_test, epochs = 60, batch_size = 64, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	f_size = max(1, int(window_size*0.25))
	k_size = max(2, int(f_size*0.1))

	model = Sequential()
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', input_shape=(n_timesteps,n_features), padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.35))
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(n_outputs, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	#es = [EarlyStopping(monitor='val_accuracy', mode='max', patience = 8, restore_best_weights=True)]
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)#, callbacks=es)

	return results.history["val_accuracy"][-1]

''' #27
[[['Conv1D', 'Dropout', 'Conv1D', 'Dropout', 'Conv1D', 'Dropout', 'Flatten'], [0.9608378410339355], [89.44924180842054], [0.9900000095367432], [16.305859088897705], [200, 0, 256]], {'Conv1D_0': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_1': {'rate': [0.35]}, 'Conv1D_2': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_3': {'rate': [0.2]}, 'Conv1D_4': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_5': {'rate': [0.5]}, 'optimizer': 'RMSprop'}]
'''
def model_cnn_27(x_train, y_train, x_test, y_test, epochs = 60, batch_size = 64, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	f_size = max(1, int(window_size*0.25))
	k_size = max(2, int(f_size*0.1))

	model = Sequential()
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', input_shape=(n_timesteps,n_features), padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.35))
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Conv1D(filters=f_size, kernel_size=k_size, activation='relu', padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(n_outputs, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	#es = [EarlyStopping(monitor='val_accuracy', mode='max', patience = 8, restore_best_weights=True)]
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)#, callbacks=es)

	return results.history["val_accuracy"][-1]
#------------------------------------------------------------------------------
#GRU

'''#7
[[['GRU', 'Dropout', 'GRU', 'GRU', 'GRU', 'GRU', 'Dropout'], [0.957575798034668], [1.111794701549742], [1.0], [3.2841466665267944], [400, 0, 16]], {'GRU_0': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dropout_1': {'rate': [0.5]}, 'GRU_2': {'units': [500], 'dropout': [0.25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'GRU_3': {'units': [250], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'GRU_4': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'GRU_5': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['relu']}, 'Dropout_6': {'rate': [0.5]}, 'optimizer': 'RMSprop'}]
'''
def model_gru_07(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(GRU(100, dropout = 0.5, bias_initializer = "RandomNormal", input_shape=(n_timesteps, n_features), return_sequences = True))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(GRU(500, dropout = 0.25, bias_initializer = "RandomNormal", return_sequences = True))
	model.add(LeakyReLU())
	model.add(GRU(250, dropout = 0.5, bias_initializer = "glorot_normal", activation = 'tanh', return_sequences = True))
	model.add(GRU(500, dropout = 0.5, bias_initializer = "Zeros", activation = 'tanh', return_sequences = True))
	model.add(GRU(100, dropout = 0.5, bias_initializer = "Zeros", activation = 'relu', return_sequences = False))
	model.add(Dropout(0.5))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#12
[[['GRU', 'GRU', 'Dropout', 'Dense', 'Dense', 'Dense', 'Dense'], [0.9716024398803711], [1.0779674326430468], [0.9722222089767456], [3.143951892852783], [400, 0, 64]], {'GRU_0': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'GRU_1': {'units': [25], 'dropout': [0], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_2': {'rate': [0.35]}, 'Dense_3': {'units': [25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_4': {'units': [100], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_5': {'units': [10], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_6': {'units': [250], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'optimizer': 'adam'}]
'''
def model_gru_12(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(GRU(100, dropout = 0.5, bias_initializer = "RandomNormal", input_shape=(n_timesteps, n_features), return_sequences = True))
	model.add(LeakyReLU())
	model.add(GRU(25, bias_initializer = "glorot_normal", activation = "tanh", return_sequences = False))
	model.add(Dropout(0.35))
	model.add(Dense(25, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(100, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(10, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(250, bias_initializer = "Zeros", activation='tanh'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#15
[[['GRU', 'GRU', 'Dense', 'Dropout', 'Dense', 'Dense', 'Dense'], [0.9481680393218994], [1.0669472916942286], [0.9482758641242981], [1.1120991110801697], [200, 0, 256]], {'GRU_0': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'GRU_1': {'units': [25], 'dropout': [0], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_2': {'units': [250], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dropout_3': {'rate': [0.35]}, 'Dense_4': {'units': [100], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dense_5': {'units': [10], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dense_6': {'units': [250], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'optimizer': 'RMSprop'}]
'''
def model_gru_15(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(GRU(100, dropout = 0.5, bias_initializer = "RandomNormal", input_shape=(n_timesteps, n_features), return_sequences = True))
	model.add(LeakyReLU())
	model.add(GRU(25, bias_initializer = "glorot_normal", activation = "tanh", return_sequences = False))
	model.add(Dense(250, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dropout(0.35))
	model.add(Dense(100, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dense(10, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dense(250, bias_initializer = "Zeros", activation='tanh'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#------------------------------------------------------------------------------
#LSTM
'''#4
[[['LSTM', 'LSTM', 'Dropout', 'Dense', 'Dense', 'Dense', 'Dropout'], [1.0], [1.1018270254135132], [1.0], [3.143376350402832], [400, 25, 256]], { 'LSTM_0': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'LSTM_1': {'units': [500], 'dropout': [0.25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dropout_2': {'rate': [0.35]}, 'Dense_3': {'units': [50], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_4': {'units': [250], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dense_5': {'units': [250], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_6': {'rate': [0.35]}, 'optimizer': 'adam'}]
'''
def model_lstm_04(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(LSTM(500, dropout = 0.5, bias_initializer = "glorot_normal", activation = "tanh", input_shape=(n_timesteps, n_features), return_sequences = True))
	model.add(LSTM(500, dropout = 0.25, bias_initializer = "RandomNormal", return_sequences = False))
	model.add(LeakyReLU())
	model.add(Dropout(0.35))
	model.add(Dense(50, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(250, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dense(250, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dropout(0.35))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#5
[[['LSTM', 'LSTM', 'Dropout', 'Dense', 'Dense', 'Dense', 'Dropout'], [0.9814814925193787], [1.1301309010128917], [1.0], [1.8104376196861267], [400, 50, 64]], {'LSTM_0': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'LSTM_1': {'units': [500], 'dropout': [0.25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dropout_2': {'rate': [0.5]}, 'Dense_3': {'units': [50], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_4': {'units': [25], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_5': {'units': [250], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_6': {'rate': [0.35]}, 'optimizer': 'RMSprop'}]
'''
def model_lstm_05(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(LSTM(500, dropout = 0.5, bias_initializer = "glorot_normal", activation = "tanh", input_shape=(n_timesteps, n_features), return_sequences = True))
	model.add(LSTM(500, dropout = 0.25, bias_initializer = "RandomNormal", return_sequences = False))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(Dense(50, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(25, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(250, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dropout(0.35))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#16
[[['LSTM', 'LSTM', 'Dropout', 'Dense', 'Dense', 'Dense', 'Dropout'], [0.9898580312728882], [0.8508668593291578], [1.0], [2.4275476101861893], [400, 0, 16]], { 'LSTM_0': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'LSTM_1': {'units': [500], 'dropout': [0.25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dropout_2': {'rate': [0.35]}, 'Dense_3': {'units': [50], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_4': {'units': [250], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dense_5': {'units': [250], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_6': {'rate': [0.35]}, 'optimizer': 'adam'}]
'''
def model_lstm_16(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(LSTM(500, dropout = 0.5, bias_initializer = "glorot_normal", activation = "tanh", input_shape=(n_timesteps, n_features), return_sequences = True))
	model.add(LSTM(500, dropout = 0.25, bias_initializer = "RandomNormal", return_sequences = False))
	model.add(LeakyReLU())
	model.add(Dropout(0.35))
	model.add(Dense(50, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(250, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dense(250, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dropout(0.35))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#------------------------------------------------------------------------------
#Dense
'''#0
[[['Dense', 'Dropout', 'Dense', 'Dropout', 'Dense'], [0.9146760702133179], [0.9374584359225665], [0.7779021859169006], [1.1499179669835837], [0, 0, 256]], {'Dense_0': {'units': [25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dropout_1': {'rate': [0.2]}, 'Dense_2': {'units': [10], 'bias_initializer': ['Zeros'], 'activation': ['relu']}, 'Dropout_3': {'rate': [0.2]}, 'Dense_4': {'units': [50], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'optimizer': 'adam'}]
'''
def model_conv1d_00(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 100, window_size = 200):
	model = Sequential()
	model.add(Dense(25, bias_initializer = "RandomNormal", input_dim=x_train.shape[1], activation='relu'))
	model.add(LeakyReLU())
	model.add(Dropout(0.2))
	model.add(Dense(10, bias_initializer = "Zeros", activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(50, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#1
[[['Dense', 'Dense', 'Dropout', 'Dense', 'Dense', 'Dropout', 'Dense'], [0.9383819699287415], [0.9200891889451024], [0.7591284513473511], [1.4496575506006875], [0, 0, 256]], {'Dense_0': {'units': [25], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'Dense_1': {'units': [10], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dropout_2': {'rate': [0.5]}, 'Dense_3': {'units': [500], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dense_4': {'units': [50], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_5': {'rate': [0.2]}, 'Dense_6': {'units': [10], 'bias_initializer': ['Zeros'], 'activation': ['LeakyReLU']}, 'optimizer': 'RMSprop'}]
'''
def model_conv1d_01(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 100, window_size = 200):
	model = Sequential()
	model.add(Dense(25, bias_initializer = "RandomNormal", input_dim=x_train.shape[1], activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(10, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(500, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dense(50, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dropout(0.2))
	model.add(Dense(10, bias_initializer = "Zeros", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#4
[[['Dense', 'Dropout', 'Dense', 'Dropout', 'Dense', 'Dense', 'Dense'], [0.9273950457572937], [0.6039403113260408], [0.7502492666244507], [1.1387167165853085], [0, 0, 16]], {'Dense_0': {'units': [250], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'Dropout_1': {'rate': [0.5]}, 'Dense_2': {'units': [25], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dropout_3': {'rate': [0.5]}, 'Dense_4': {'units': [500], 'bias_initializer': ['Zeros'], 'activation': ['LeakyReLU']}, 'Dense_5': {'units': [100], 'bias_initializer': ['RandomNormal'], 'activation': ['relu']}, 'Dense_6': {'units': [10], 'bias_initializer': ['Zeros'], 'activation': ['LeakyReLU']}, 'optimizer': 'adam'}]
'''
def model_conv1d_04(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 100, window_size = 200):
	model = Sequential()
	model.add(Dense(250, bias_initializer = "Zeros", input_dim=x_train.shape[1], activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(25, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(500, bias_initializer = "Zeros", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(100, bias_initializer = "RandomNormal", activation='relu'))
	model.add(Dense(10, bias_initializer = "Zeros", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#------------------------------------------------------------------------------
#ConvLSTM2D
'''#0
[[['ConvLSTM2D', 'Dropout', 'MaxPooling1D', 'ConvLSTM2D', 'Dropout', 'ConvLSTM2D', 'Flatten'], [0.9938271641731262], [1.2244070172309875], [1.0], [1.8456576466560364], [400, 25, 256]], { 'ConvLSTM2D_0': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_1': {'rate': [0.5]}, 'MaxPooling1D_2': {'pool_size': [0.2]}, 'ConvLSTM2D_3': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_4': {'rate': [0.35]}, 'ConvLSTM2D_5': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'optimizer': 'RMSprop'}]
'''
def model_convlstm2d_00(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 8, window_size = 200):
	f_size = max(1, int(0.25*window_size))
	k_size = max(2, int(0.1*window_size))
	p_size = max(1, int(0.2*window_size))

	model = Sequential()
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same",
						 input_shape=(x_train.shape[1], x_train.shape[2],
									  x_train.shape[3], x_train.shape[4])))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(MaxPooling3D(pool_size=(1, p_size, p_size), padding="same"))
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.35))
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = False, padding="same"))
	model.add(LeakyReLU())
	model.add(Flatten())
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	# fit network
	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#14
[[['ConvLSTM2D', 'Dropout', 'MaxPooling1D', 'ConvLSTM2D', 'Dropout', 'ConvLSTM2D', 'Flatten'], [0.9938271641731262], [1.2154372757599679], [1.0], [3.6911762952804565], [400, 50, 64]], {'ConvLSTM2D_0': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_1': {'rate': [0.5]}, 'MaxPooling1D_2': {'pool_size': [0.1]}, 'ConvLSTM2D_3': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_4': {'rate': [0.35]}, 'ConvLSTM2D_5': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'optimizer': 'adam'}]
'''
def model_convlstm2d_14(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 8, window_size = 200):
	f_size = max(1, int(0.25*window_size))
	k_size = max(2, int(0.1*window_size))
	p_size = max(1, int(0.1*window_size))

	model = Sequential()
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same",
						 input_shape=(x_train.shape[1], x_train.shape[2],
									  x_train.shape[3], x_train.shape[4])))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(MaxPooling3D(pool_size=(1, p_size, p_size), padding="same"))
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.35))
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same"))
	model.add(LeakyReLU())
	model.add(Flatten())
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#36
[[['ConvLSTM2D', 'MaxPooling1D', 'ConvLSTM2D', 'Dropout', 'ConvLSTM2D', 'MaxPooling1D', 'Flatten'], [0.9937238693237305], [1.0253959674715496], [1.0], [1.3666332275827175], [400, 0, 16]], {'ConvLSTM2D_0': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'MaxPooling1D_1': {'pool_size': [0.2]}, 'ConvLSTM2D_2': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'Dropout_3': {'rate': [0.2]}, 'ConvLSTM2D_4': {'activation': ['LeakyReLU'], 'filters': [0.25]}, 'MaxPooling1D_5': {'pool_size': [0.1]}, 'optimizer': 'adam'}]
'''
def model_convlstm2d_36(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 8, window_size = 200):
	f_size = max(1, int(0.25*window_size))
	k_size = max(2, int(0.1*window_size))
	p_size1 = max(1, int(0.2*window_size))
	p_size2 = max(1, int(0.1*window_size))

	model = Sequential()
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same",
						 input_shape=(x_train.shape[1], x_train.shape[2],
									  x_train.shape[3], x_train.shape[4])))
	model.add(LeakyReLU())
	model.add(MaxPooling3D(pool_size=(1, p_size1, p_size1), padding="same"))
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = True, padding="same"))
	model.add(LeakyReLU())
	model.add(Dropout(0.5))
	model.add(ConvLSTM2D(filters=f_size, kernel_size=(1, k_size), activation='relu',
					     return_sequences = False, padding="same"))
	model.add(LeakyReLU())
	model.add(MaxPooling2D(pool_size=(1, p_size2), padding="same"))
	model.add(Flatten())
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#------------------------------------------------------------------------------
#BidirectionalLSTM

'''#2
[[['BidirectionalLSTM', 'Dropout', 'BidirectionalLSTM', 'Dense'], [1.0], [1.1105656623840332], [1.0], [1.4104014933109283], [200, 50, 256]], {'BidirectionalLSTM_0': {'units': [500], 'dropout': [0.25], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_1': {'rate': [0.5]}, 'BidirectionalLSTM_2': {'units': [100], 'dropout': [0], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_3': {'units': [250], 'bias_initializer': ['RandomNormal'], 'activation': ['LeakyReLU']}, 'optimizer': 'adam'}]
'''
def model_bidirlstm_02(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Bidirectional(LSTM(500, dropout = 0.25, bias_initializer = "glorot_normal",
							  activation = "tanh", return_sequences = True,
							  input_shape=(n_timesteps, n_features))))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(100, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = False)))
	model.add(Dense(250, bias_initializer = "RandomNormal", activation='relu'))
	model.add(LeakyReLU())
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#14
[[['BidirectionalLSTM', 'Dropout', 'BidirectionalLSTM', 'Dense', 'Dense', 'Dense'], [0.9877300262451172], [1.0045039524573733], [1.0], [0.9942757209593599], [400, 0, 16]], {'BidirectionalLSTM_0': {'units': [250], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_1': {'rate': [0.5]}, 'BidirectionalLSTM_2': {'units': [25], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_3': {'units': [100], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_4': {'units': [250], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'Dense_5': {'units': [50], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'optimizer': 'RMSprop'}]
'''
def model_bidirlstm_14(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Bidirectional(LSTM(250, dropout = 0.5, bias_initializer = "glorot_normal",
							  activation = "tanh", return_sequences = True,
							  input_shape=(n_timesteps, n_features))))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(25, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = False)))
	model.add(Dense(100, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(250, bias_initializer = "Zeros", activation='tanh'))
	model.add(Dense(50, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#27
[[['BidirectionalLSTM', 'Dropout', 'BidirectionalLSTM', 'Dense', 'Dense', 'Dense'], [0.9865951538085938], [0.9527801211067776], [0.982758641242981], [0.8290504515171051], [200, 0, 64]], {'BidirectionalLSTM_0': {'units': [250], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_1': {'rate': [0.5]}, 'BidirectionalLSTM_2': {'units': [25], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_3': {'units': [100], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dense_4': {'units': [250], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'Dense_5': {'units': [50], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'optimizer': 'RMSprop'}]
'''
def model_bidirlstm_27(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Bidirectional(LSTM(250, dropout = 0.5, bias_initializer = "glorot_normal",
							  activation = "tanh", return_sequences = True,
							  input_shape=(n_timesteps, n_features))))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(25, dropout = 0.5, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = False)))
	model.add(Dense(100, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(250, bias_initializer = "Zeros", activation='tanh'))
	model.add(Dense(50, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#------------------------------------------------------------------------------
#BidirectionalGRU

'''#7
[[['BidirectionalGRU', 'Dropout', 'BidirectionalGRU', 'BidirectionalGRU', 'BidirectionalGRU'], [0.9259259700775146], [1.8456452290217082], [1.0], [3.6180601119995117], [200, 25, 16]], {'BidirectionalGRU_0': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'Dropout_1': {'rate': [0.35]}, 'BidirectionalGRU_2': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['LeakyReLU']}, 'BidirectionalGRU_3': {'units': [100], 'dropout': [0.25], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'BidirectionalGRU_4': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'optimizer': 'RMSprop'}]
'''
def model_bidirgru_07(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Bidirectional(GRU(500, dropout = 0.5, bias_initializer = "Zeros",
							  activation = "tanh", return_sequences = True,
							  input_shape=(n_timesteps, n_features))))
	model.add(Dropout(0.35))
	model.add(Bidirectional(GRU(500, dropout = 0.5, bias_initializer = "Zeros",
							     activation = "relu", return_sequences = True)))
	model.add(LeakyReLU())
	model.add(Bidirectional(GRU(100, dropout = 0.25, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = True)))
	model.add(Bidirectional(GRU(100, dropout = 0.5, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = False)))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#14
[[['BidirectionalGRU', 'Dropout', 'BidirectionalGRU', 'BidirectionalGRU', 'BidirectionalGRU'], [0.9074074029922485], [1.1804003877404297], [1.0], [1.3083578944206238], [400, 25, 64]], {'BidirectionalGRU_0': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'Dropout_1': {'rate': [0.35]}, 'BidirectionalGRU_2': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['LeakyReLU']}, 'BidirectionalGRU_3': {'units': [100], 'dropout': [0.25], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'BidirectionalGRU_4': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'optimizer': 'adam'}]
'''
def model_bidirgru_14(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Bidirectional(GRU(500, dropout = 0.5, bias_initializer = "Zeros",
							  activation = "tanh", return_sequences = True,
							  input_shape=(n_timesteps, n_features))))
	model.add(Dropout(0.35))
	model.add(Bidirectional(GRU(500, dropout = 0.5, bias_initializer = "Zeros",
							     activation = "relu", return_sequences = True)))
	model.add(LeakyReLU())
	model.add(Bidirectional(GRU(100, dropout = 0.25, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = True)))
	model.add(Bidirectional(GRU(100, dropout = 0.5, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = False)))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

'''#19
[[['BidirectionalGRU', 'BidirectionalGRU', 'BidirectionalGRU', 'BidirectionalGRU', 'BidirectionalGRU', 'Dropout', 'Dense'], [0.9900362491607666], [1.750424323738485], [0.9886363744735718], [1.1907527446746826], [200, 0, 256]], {'BidirectionalGRU_0': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['tanh']}, 'BidirectionalGRU_1': {'units': [100], 'dropout': [0], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'BidirectionalGRU_2': {'units': [500], 'dropout': [0.5], 'bias_initializer': ['Zeros'], 'activation': ['LeakyReLU']}, 'BidirectionalGRU_3': {'units': [100], 'dropout': [0.25], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'BidirectionalGRU_4': {'units': [100], 'dropout': [0.5], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'Dropout_5': {'rate': [0.5]}, 'Dense_6': {'units': [50], 'bias_initializer': ['glorot_normal'], 'activation': ['tanh']}, 'optimizer': 'RMSprop'}]
'''
def model_bidirgru_19(x_train, y_train, x_test, y_test, epochs = 10, batch_size = 4, window_size = 200):
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

	model = Sequential()
	model.add(Bidirectional(GRU(500, dropout = 0.5, bias_initializer = "Zeros",
							  activation = "tanh", return_sequences = True,
							  input_shape=(n_timesteps, n_features))))
	model.add(Bidirectional(GRU(100, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = True)))
	model.add(Bidirectional(GRU(500, dropout = 0.5, bias_initializer = "Zeros",
							     activation = "relu", return_sequences = True)))
	model.add(LeakyReLU())
	model.add(Bidirectional(GRU(100, dropout = 0.25, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = True)))
	model.add(Bidirectional(GRU(100, dropout = 0.5, bias_initializer = "glorot_normal",
							     activation = "tanh", return_sequences = False)))
	model.add(Dropout(0.5))
	model.add(Dense(50, bias_initializer = "glorot_normal", activation='tanh'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)

	return results.history["val_accuracy"][-1]

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#START| MAIN

#pamap2      = 100hz
#har         = 50hz
#firebusters = 400hz
"""
#==============================================================================
#BidirectionalGRU MODELS 7, 14, 19
#------------------------------------------------------------------------------
#PAMAP2
#---7--- OOM?
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 50			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirgru_07, model_params, 2)

#---14---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 100			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirgru_14, model_params, 2)

#---19---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 50			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirgru_19, model_params, 2)

print("===========BidirectionGRU PAMAP2 Done!===========")
#------------------------------------------------------------------------------
#HAR
#---7--- OOM?
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 25			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirgru_07, model_params, 2)

#---14---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirgru_14, model_params, 2)

#---19---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 25			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirgru_19, model_params, 2)

print("===========BidirectionGRU HAR Done!===========")

#==============================================================================
#BidirectionalLSTM MODELS 2, 14, 27
#------------------------------------------------------------------------------
#PAMAP2
#---2---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 50,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 50			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirlstm_02, model_params, 2)

#---14---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 100			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirlstm_14, model_params, 2)

#---27--- OOM?
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirlstm_27, model_params, 2)

print("===========BidirectionLSTM PAMAP2 Done!===========")

#------------------------------------------------------------------------------
#HAR
#---2---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 50,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 25			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirlstm_02, model_params, 2)

#---14---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 50			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirlstm_14, model_params, 2)

#---27--- OOM?
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,				#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 25			#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_bidirlstm_27, model_params, 2)

print("===========BidirectionLSTM HAR Done!===========")
"""
#==============================================================================
#ConvLSTM2D MODELS 0, 14, 36
#------------------------------------------------------------------------------
#PAMAP2
#---0---
lay_gen = Layer_Generator()
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 100			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_convlstm2d_00, model_params, 2)

#---14---
lay_gen = Layer_Generator()
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 50,
			   'LOSO' : True,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 100			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_convlstm2d_14, model_params, 2)

#---36---
lay_gen = Layer_Generator()
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 100			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_convlstm2d_36, model_params, 2)

print("=========== ConvLSTM2D PAMAP2 Done!===========")
#------------------------------------------------------------------------------
#HAR
#---0---
lay_gen = Layer_Generator()
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 50			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_convlstm2d_00, model_params, 2)

#---14---
lay_gen = Layer_Generator()
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 50,
			   'LOSO' : True,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_convlstm2d_14, model_params, 2)

#---36---
lay_gen = Layer_Generator()
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,				#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : clstm_params
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 50			#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_convlstm2d_36, model_params, 2)

print("=========== ConvLSTM2D HAR Done!===========")
"""
#==============================================================================
#Dense MODELS 0, 1, 4
#------------------------------------------------------------------------------
#PAMAP2
#---0---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 0,
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 0
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_conv1d_00, model_params, 2)

#---1---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 0,
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 0
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_conv1d_01, model_params, 2)

#---4---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 0,
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 0
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_conv1d_04, model_params, 2)

print("=========== Dense PAMAP2 Done!===========")
#------------------------------------------------------------------------------
#HAR
#---0---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 0,
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 0
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_conv1d_00, model_params, 2)

#---1---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 0,
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 0
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_conv1d_01, model_params, 2)

#---4---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 0,
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 0
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_conv1d_04, model_params, 2)

print("=========== Dense HAR Done!===========")
#==============================================================================
#LSTM MODELS 4, 5, 16
#------------------------------------------------------------------------------
#PAMAP2
#---4---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 100		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_lstm_04, model_params, 2)

#---5---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 50,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 100		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_lstm_05, model_params, 2)

#---16--- OOM?
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 100		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_lstm_04, model_params, 2)

print("=========== LSTM PAMAP2 Done!===========")
#------------------------------------------------------------------------------
#HAR
#---4---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 50		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_lstm_04, model_params, 2)

#---5---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 50,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_lstm_05, model_params, 2)

#---16--- OOM?
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 50		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_lstm_04, model_params, 2)

print("=========== LSTM HAR Done!===========")
#==============================================================================
#GRU MODELS 7, 12, 15
#------------------------------------------------------------------------------
#PAMAP2
#---7--- OOM?
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 100		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_07, model_params, 2)

#---12---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 100		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_12, model_params, 2)

#---15---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,			#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 50		#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_15, model_params, 2)

print("=========== GRU PAMAP2 Done!===========")
#------------------------------------------------------------------------------
#HAR
#---7--- OOM?
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 16,
				"window_size" : 50		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_07, model_params, 2)

#---12---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_12, model_params, 2)

#---15---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,			#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 25		#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_gru_15, model_params, 2)

print("=========== GRU HAR Done!===========")

#==============================================================================
#Conv1D MODELS 5, 16, 27
#------------------------------------------------------------------------------
#PAMAP2 Conv1D

#dataset, train_p = 0.8, w_size = 0, o_percent = 0.25):
#---5---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,			#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0.25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50 	    #200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
			   model_cnn_05, model_params, 2)

#---16---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 100,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 100		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_cnn_16, model_params, 2)

#---27---
data_params = {'dataset' : 'pamap2',
               'train_p' : 0.96,
               'w_size' : 50,			#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 50		#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_cnn_27, model_params, 2)

print("=========== Conv1D PAMAP2 Done!===========")
#------------------------------------------------------------------------------
#HAR Conv1D

#dataset, train_p = 0.8, w_size = 0, o_percent = 0.25):
#---5---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,			#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0.25,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 25 	    #200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
			   model_cnn_05, model_params, 2)

#---16---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 50,			#400:firebusters, 50:har, 100:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 64,
				"window_size" : 50		#400:firebusters, 50:har, 100:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_cnn_16, model_params, 2)

#---27---
data_params = {'dataset' : 'har',
               'train_p' : 0.96,
               'w_size' : 25,			#200:firebusters, 25:har, 50:pamap2
               'o_percent' : 0,
			   'LOSO' : True,
               'clstm_params' : {}
               }
dataset = Load_Data(**data_params)
model_params = {'epochs' : 60,
                'batch_size' : 256,
				"window_size" : 25		#200:firebusters, 25:har, 50:pamap2
                }
run_experiment(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test,
                model_cnn_27, model_params, 2)

print("=========== Conv1D HAR Done!===========")
#==============================================================================









"""









