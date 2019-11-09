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
from keras.layers import Embedding
from keras.utils import to_categorical
import itertools
import random

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

#------------------------------------------------------------------------------
#START| LOADING FIREBUSTERS DATASET
    
def load_dataset_windows(df, t_window = 200, t_overlap = 0.25):
    #get all exercise ID's
    df_exid = df['exercise_id'].unique()
    #get all subject ID's
    df_subid = df['subject_id'].unique()
    #get all session ID's
    df_sesid = df['session_id'].unique()
    
    #This makes all possible combinations of session/exercise/subject
    all_combo = list(itertools.product(df_exid, df_subid, df_sesid))
    
    #This makes a separate dataframe of each session/exercise/subject combination
    df_all = []
    for combo in all_combo:
        if ((df['exercise_id'] == combo[0]) & (df['subject_id'] == combo[1]) & (df['session_id'] == combo[2])).any():
            #This combination exists, get all rows that match this
           df_all.append( df.loc[(df['exercise_id'] == combo[0]) & (df['subject_id'] == combo[1]) & (df['session_id'] == combo[2])] )

    #Makes the windows
    windows = []
    for a_df in df_all:
        #Number of rows in the dataframe
        nrows = a_df.shape[0]
        #Number of windows in the dataframe
        num_windows = int(( (nrows-t_window)/(t_window*(1-t_overlap)) )+1)
        #The starting offset (ex: t_window:200 t_overlap:0.25 offset:150)
        offset = int(t_window*(1-t_overlap))
        #Number of rows used from the dataframe, used to determine the last
        #locations windows to start at
        rows_used = int(t_window + (num_windows-2)*offset)
        
        #This puts the first window dataframe into the list
        if rows_used >= t_window:
            windows.append(a_df[0:t_window].to_numpy())
        
        #Puts the remaining windows into the list
        for i in range(offset, rows_used, offset):
            windows.append( a_df[i:i+t_window].to_numpy() )
    
    windows = np.array(windows)
    
    y_axis = df.columns.get_loc("exercise_id")

    #Delete columns we don't want
    cols_to_delete = []
    cols_to_delete.append(df.columns.get_loc("TimeStamp_s"))
    cols_to_delete.append(df.columns.get_loc("exercise_amt"))
    cols_to_delete.append(df.columns.get_loc("session_id"))
    cols_to_delete.append(df.columns.get_loc("subject_id"))
    cols_to_delete.append(y_axis)
    
    y = windows[:, :, y_axis] - 1
    y = to_categorical(y)
    
    x = np.copy(windows)
    x = np.delete(x, cols_to_delete, axis = 2)
    
    #Train size
    frac = 0.8
    train_size = int(x.shape[0]*frac)
    train_indices = list(random.sample(range(0, x.shape[0]), train_size))
    all_values = np.arange(0, windows.shape[0])
    test_indices = [ti for ti in all_values if ti not in train_indices]
    
    x_train = np.array([x[i,:,:] for i in train_indices])
    y_train = np.array([y[i,:,:] for i in train_indices])
    x_test = np.array([x[i,:,:] for i in test_indices])
    y_test = np.array([y[i,:,:] for i in test_indices])
    
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train, x_test, y_test)

# load the dataset, returns train and test X and y elements
def load_dataset():
	df = pd.read_csv("Zenshin_Data/ComboPlatter.csv")
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

#------------------------------------------------------------------------------
#START| MODEL STUFFS

#Fit and evaluate a model
def evaluate_model(x_train, y_train, x_test, y_test):
	epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	model = Sequential()
	#model.add(Embedding(57, 32, input_shape=(len(x_train), 200, 57)))
	model.add(LSTM(32))
	model.add(Dense(32, input_dim=205, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	# model.add(Dense(y_train.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Define model
	# model = Sequential()
	# model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length)))
	# model.add(Dropout(0.5))
	# model.add(Flatten())
	# model.add(Dense(100, activation='relu'))
	# model.add(Dense(n_outputs, activation='softmax'))
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit network
	model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

	# evaluate model
	_, accuracy = model.evaluate(x_test, y_test, batch_size = batch_size, verbose=verbose)

	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=1):
	# load data
	df = pd.read_csv("ComboPlatter.csv") 
	x_train, y_train, x_test, y_test = load_dataset_windows(df)
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(x_train, y_train, x_test, y_test)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

#------------------------------------------------------------------------------
#START| MAIN

# run the experiment
#run_experiment(1)

#x_train, y_train, x_test, y_test = load_dataset()
#print(x_test.columns)#shape, y_train.shape, x_test.shape, y_test.shape)
#n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
#print(x_train.shape)

#print("Exercise_id, subject_id, session_id")

run_experiment(repeats=1)

#print(x_test.shape, y_train.shape, x_test.shape, y_test.shape)

#print(val)






































# def split_window(df,t_window = 200 ):
# 	# reshape input to be 3D [samples, timesteps, features]
# 	# Timestep = 2 seconds.  200 rows
# 	# Thought process
# 		# create empty window list
# 		# specify window depth
# 		# Isolate each subject and activity in a for loop

# 	# split into samples (e.g. train_x = 31382/200 = 156 time windows)
# 	# step over the data range in jumps of 200
# 	print(df.shape)

# 	# Makes the windows, non-overlaping
# 	windows = list()
# 	items_in_window = []
# 	window_counter = 0
# 	for i in range(0,df.shape[0]):
# 		items_in_window.append(df.iloc[i])
# 		window_counter += 1
# 		if window_counter >= t_window:
# 			window_counter = 0
# 			windows.append(item_in_window)
# 			item_in_window = []

# 	#samples = dstack(samples)
# 	#print(samples.shape)

# 	#data = data.reshape((len(samples), n_steps, 57))
# 	#print(data.shape)
# 	return data

# def split_df(X,y,split=0.2):

# 	#Split data into test and train for attributes and target column
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

# 	# offset class values from zero
# 	y_train = y_train - 1
# 	y_test = y_test - 1

# 	# one hot encode y
# 	y_train = to_categorical(y_train)
# 	y_test = to_categorical(y_test)

	

# 	print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# 	return X_train, X_test, y_train, y_test

# def process_drops(df, cols):
# 	return df.drop(cols,axis=1,inplace=True)			 			#Drops columns

# # def process_window(df):
# #     #This function will assign time windows eventually

# def build_df(drops=["exercise_amt"]):
# 	df = pd.read_csv("ComboPlatter.csv")  											#Load Data
# 	process_drops(df,drops)															#Drop columns
# 	# process_window(df)
# 	y = df["exercise_id"]															#Split out exercise_id
# 	X = df.drop(columns=['exercise_id'])
# 	return X,y,df

# def evaluate_model(X_train, X_test, y_train, y_test):
# 	verbose, epochs, batch_size = 0, 15, 64
# 	n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
# 	model = Sequential()
# 	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(100, activation='relu'))
# 	model.add(Dense(n_outputs, activation='softmax'))
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# fit network
# 	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# 	# evaluate model
# 	_, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
# 	return accuracy
# # summarize scores
# def summarize_results(scores):
# 	print(scores)
# 	m, s = mean(scores), std(scores)
# 	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# # run an experiment
# def run_experiment(repeats=1):
# 	# load data
# 	X,y,df = build_df()
# 	df = split_window(df)
# 	X_train, X_test, y_train, y_test = split_df(X,y,0.2)
# 	# repeat experiment
# 	scores = list()
# 	for r in range(repeats):
# 		score = evaluate_model(X_train, X_test, y_train, y_test)
# 		score = score * 100.0
# 		print('>#%d: %.3f' % (r+1, score))
# 		scores.append(score)
# 	# summarize results
# 	summarize_results(scores)
 
# # run the experiment
# run_experiment()
# # X,y,df = build_df()
# # X_train, X_test, y_train, y_test = split_df(X,y,0.2)
# # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# plot_accel(1, 1, 1, df)


