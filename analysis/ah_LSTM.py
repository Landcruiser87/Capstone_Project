#%%
# ms-python.python added
import os
try:
	os.chdir('~/CapstoneA') 
	print(os.getcwd())
except:
	pass

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import array
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical


os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")

def plot_accel(activity, subject, session, data):
	data = data.iloc[500:1000]   									#Select a certain time period
	data = data[data['subject_id']==subject] 						#Filter by person
	data = data[data['session_id']==session] 						#Filter by session
	data = data[data['exercise_id']==activity] 						#filter by exercise

	#Graph all 3 axis on the same plot
	plt.plot( 'TimeStamp_s', 'sID1_AccX_g', data=data, marker='', color='skyblue', linewidth=2, label='X-axis')
	plt.plot( 'TimeStamp_s', 'sID1_AccY_g', data=data, marker='', color='olive', linewidth=2, label="Y-Axis")
	plt.plot( 'TimeStamp_s', 'sID1_AccZ_g', data=data, marker='', color='brown', linewidth=2, label="Z-Axis")
	plt.legend()

def split_sequences(sequences, n_steps):
	# split into samples (e.g. 5000/200 = 25)
	# step over the data range in jumps of 200
	print(sequences.shape)
	samples = list()
	for i in range(0,len(sequences),n_steps):
		# grab from i to i + 200
		sample = sequences[i:i+n_steps]
		samples.append(sample)
	data = array(samples)
	print(data.shape)
	data = data.reshape((len(samples), n_steps, 57))
	print(data.shape)
	return data

def split_df(X,y,split=0.2):
	#Split data into test and train for attributes and target column
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

	# offset class values from zero
	y_train = y_train - 1
	y_test = y_test - 1

	# one hot encode y
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	
	# reshape input to be 3D [samples, timesteps, features]
	# Timestep = 2 seconds.  200 rows
	# Fuck i dont' know how to do this
	t_window = 200
	X_train = split_sequences(X_train, t_window)
	X_test = split_sequences(X_test, t_window)
	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
	return X_train, X_test, y_train, y_test

def process_drops(df, cols):
	return df.drop(cols,axis=1,inplace=True)			 			#Drops columns

# def process_window(df):
#     #This function will assign time windows eventually

def build_df(drops=["exercise_amt"]):
	df = pd.read_csv("ComboPlatter.csv")  											#Load Data
	process_drops(df,drops)															#Drop columns
	# process_window(df)
	y = df["exercise_id"]															#Split out exercise_id
	X = df.drop(columns=['TimeStamp_s','exercise_id','subject_id','session_id']) 	#Load rest of attributes into new dataframe X
	return X,y,df

def evaluate_model(X_train, X_test, y_train, y_test):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
	return accuracy
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=1):
	# load data
	X,y,df = build_df()
	X_train, X_test, y_train, y_test = split_df(X,y,0.2)
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(X_train, X_test, y_train, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment
run_experiment()
# X,y,df = build_df()
# X_train, X_test, y_train, y_test = split_df(X,y,0.2)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# plot_accel(1, 1, 1, df)


