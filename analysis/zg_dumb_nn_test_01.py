from numpy import mean
from numpy import std
from numpy import dstack
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split 

import os
os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")
import warnings
warnings.filterwarnings("ignore")

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	df = pd.read_csv("ComboPlatter.csv")
	df = df.drop(["TimeStamp_s", "exercise_amt", "session_id", "subject_id"], axis = 1)
	
	x_train = df.sample(frac = 0.8, random_state = 0)
	x_test = df.drop(x_train.index)
	y_train = x_train.pop('exercise_id')
	y_test = x_test.pop('exercise_id')
	
	#Lables start at 1, so we will subtract 1 from the y's so they start at 0
	y_train = y_train-1
	y_test = y_test-1
	
	#Makes them categorical, (one hot encoded over 3 columns)
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

	#return x_train, y_train, x_test, y_test
	return x_train.to_numpy(), y_train, x_test.to_numpy(), y_test
 
#Fit and evaluate a model
def evaluate_model(x_train, y_train, x_test, y_test):
	epochs, batch_size = 25, 64
	
	model = Sequential()
	#model.add(Embedding(57, 32, input_length = 57))
	#model.add(LSTM(32))
	model.add(Dense(32, input_dim=57, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	#Define model
	#model = Sequential()
	#model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length)))
	#model.add(Dropout(0.5))
	#model.add(Flatten())
	#model.add(Dense(100, activation='relu'))
	#model.add(Dense(n_outputs, activation='softmax'))
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
def run_experiment(repeats=5):
	# load data
	x_train, y_train, x_test, y_test = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(x_train, y_train, x_test, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
 
# run the experiment
run_experiment(1)

#x_train, y_train, x_test, y_test = load_dataset()
#print(x_test.columns)#shape, y_train.shape, x_test.shape, y_test.shape)
#n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
#print(x_train.shape)












	