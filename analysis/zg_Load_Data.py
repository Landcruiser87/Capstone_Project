"""
		LOAD HUMAN MOVEMENT DATASET

	This class loads whichever of the three
	datasets that are used in the firebusters
	exercise analysis.
	
	Parameters such as which dataset, the
	window size, and window overlap amount
	are available to change. This allows for
	ease of training the data in the other
	python files.
	
"""

from numpy import dstack
import numpy as np
import pandas as pd
from pandas import read_csv
from tensorflow.keras.utils import to_categorical
import itertools
import random

import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
#os.chdir("C:/githubrepo/CapstoneA/data/") #Zack and Andy's github data folder
import warnings
warnings.filterwarnings("ignore")

class Load_Data:
	#Initializes the class with the desired parameters
	#dataset: a string indicating which of the three datasets to load
	#w_size: indicates the window size, no windows if it equals 0
	#o_percent: overlap percentage for the windows
	def __init__(self, dataset, train_p = 0.8, LOSO = False, w_size = 0, o_percent = 0.25, clstm_params = {}, valIndex = None):
		self.dataset = dataset
		self.train_percent = train_p
		self.bwindows = False
		if w_size > 0:
			self.bwindows = True
		self.window_size = w_size
		self.overlap_percent = o_percent
		self.clstm_params = clstm_params
		self.loso = LOSO
		self.val_index = valIndex
		self.x_train, self.y_train, self.x_test, self.y_test = self.load_dataset()
		
		#In the case of this being a ConvLSTM2D
		if clstm_params != {}:
			n_steps = clstm_params["n_steps"][0]
			n_length = int(self.x_train[0].shape[1]/n_steps)
			if (self.window_size%n_steps) != 0:
				n_steps = 5
			n_features = self.x_train[0].shape[2]
			self.x_test = self.x_test.reshape((self.x_test.shape[0], n_steps, 1, n_length, n_features))
			if self.val_index == None:
				n_features = self.x_train.shape[2]
				# reshape into subsequences (samples, time steps, rows, cols, channels)
				n_length = int(self.x_train.shape[1]/n_steps)
				self.x_train = self.x_train.reshape((self.x_train.shape[0], n_steps, 1, n_length, n_features))
			else: #THIS IS WHERE WE HAVE THE CROSSVALIDATION
				print("CV reshape for ConvLSTM2D is later")

		if self.val_index == None:
			print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)
		return

	def load_dataset(self):
		if self.dataset == 'har':
			print("Using the HAR dataset")
			return self.load_dataset_har()
		elif self.dataset == 'pamap2':
			print("Using the PAMAP2 dataset")
			return self.load_dataset_pamap2()
		elif self.dataset == 'firebusters':
			print("Using the Firebusters dataset")
			return self.load_dataset_firebusters()
		else:
			print("Invalid name of dataset")
			
	def load_dataset_har(self):
		return self.load_dataset_har_initial()
	
	def load_dataset_pamap2(self):
		if self.bwindows:
			return self.load_dataset_pamap2_windows(self.window_size, self.overlap_percent)
		return self.load_dataset_pamap2_nowindows()
	
	def load_dataset_firebusters(self):
		if self.bwindows:
			if self.val_index != None:
				return self.load_dataset_fb_windows_folds(self.window_size, self.overlap_percent)
			else:
				return self.load_dataset_fb_windows(self.window_size, self.overlap_percent)
		else:
			if self.val_index != None:
				return self.load_dataset_fb_nowindows_folds()
		return self.load_dataset_fb_nowindows()

#------------------------------------------------------------------------------
#START| LOADING HAR DATASET
	def load_dataset_har_initial(self):

		if self.bwindows:
			return self.load_dataset_har_windows(self.window_size, self.overlap_percent)
		
		#START UNWINDOWING THE HAR DATASET
		trainX_w, trainY_w, testX_w, testY_w = self.har_load_dataset()
		
		# remove overlap
		cut_train = int(trainX_w.shape[1] / 2)
		cut_test = int(testX_w.shape[1] / 2)
		trainX = trainX_w[:, -cut_train:, :]
		testX = testX_w[:, -cut_test:, :]

		# flatten windows
		trainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
		testX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
		trainY = self.grow_array(trainY_w, cut_test)
		testY = self.grow_array(testY_w, cut_test)
		#print(trainX.shape, trainY.shape, testX.shape, testY.shape)
		#END UNWINDOWING THE HAR DATASET
		
		M = (trainX, trainY, testX, testY)
		return self.load_dataset_har_nowindows(M)

	def load_dataset_har_windows(self, t_window, t_overlap):
		#trainX, trainY, testX, testY = M
		
		#Make the windows
		#Need to first find out which subjects are tied to which row
		
		print("HAR with custom Windows not implemented")
		print("HAR with LOSO not implemented")
		return self.har_load_dataset()

	def load_dataset_har_nowindows(self, M):
		print("HAR NoWindows: Can't change HAR train/test split")
		return M

	#Makes the Y match the X unrolled windows
	def grow_array(self, y, w_size):
		all_rows = []
		for i in np.arange(y.shape[0]):
			for j in np.arange(w_size):
				all_rows.append(y[i,:])

		return np.array(all_rows)

	# load a single file as a numpy array
	def har_load_file(self, filepath):
		dataframe = read_csv(filepath, header=None, delim_whitespace=True)
		return dataframe.values
	 
	# load a list of files and return as a 3d numpy array
	def har_load_group(self, filenames, prefix=''):
		loaded = list()
		for name in filenames:
			data = self.har_load_file(prefix + name)
			loaded.append(data)
		# stack group so that features are the 3rd dimension
		loaded = dstack(loaded)
		return loaded
	 
	# load a dataset group, such as train or test
	def har_load_dataset_group(self, group, prefix=''):
		filepath = prefix + group + '/Inertial Signals/'
		# load all 9 files as a single array
		filenames = list()
		# total acceleration
		filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
		# body acceleration
		filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
		# body gyroscope
		filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
		# load input data
		X = self.har_load_group(filenames, filepath)
		# load class output
		y = self.har_load_file(prefix + group + '/y_'+group+'.txt')
		return X, y
	 
	# load the dataset, returns train and test X and y elements
	def har_load_dataset(self):
		#print("Fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window)")
		# load all train
		trainX, trainy = self.har_load_dataset_group('train', 'data/HAR_Dataset/')
		# load all test
		testX, testy = self.har_load_dataset_group('test', 'data/HAR_Dataset/')
		# zero-offset class values
		trainy = trainy - 1
		testy = testy - 1
		# one hot encode y
		trainy = to_categorical(trainy)
		testy = to_categorical(testy)
		return trainX, trainy, testX, testy

#------------------------------------------------------------------------------
#START| LOADING PAMAP2 DATASET
	def load_dataset_pamap2_windows(self, t_window, t_overlap):
		df = pd.read_csv("data/PAMAP2_Dataset/Protocol/ComboPlatter.csv")
				
		#Need to make activities go from 0-12 for the to_categorical function
		df.loc[df['activityID'] == 12, 'activityID'] = 8
		df.loc[df['activityID'] == 13, 'activityID'] = 9
		df.loc[df['activityID'] == 16, 'activityID'] = 10
		df.loc[df['activityID'] == 17, 'activityID'] = 11
		df.loc[df['activityID'] == 24, 'activityID'] = 12
		
		#get all exercise ID's
		df_exid = df['activityID'].unique()
		#get all subject ID's
		df_subid = df['subject_id'].unique()
		
		#This makes all possible combinations of session/exercise/subject
		all_combo = list(itertools.product(df_exid, df_subid))
		
		#This makes a separate dataframe of each session/exercise/subject combination
		df_all = []
		for combo in all_combo:
			if ((df['activityID'] == combo[0]) & (df['subject_id'] == combo[1])).any():
				#This combination exists, get all rows that match this
			   df_all.append( df.loc[(df['activityID'] == combo[0]) & (df['subject_id'] == combo[1])] )

		rand_subject = -1
		if self.loso == True:
			while True:
				rand_subject = int(random.choice(df_subid))
				#This makes sure that the subject has all 13 activities
				if len((df[df["subject_id"] == rand_subject])['activityID'].unique()) == 13:
					break

		loso_windows = []
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
			
			sid = int(a_df["subject_id"].iloc[0])
			if (self.loso == False) or (sid != rand_subject):			
				#This puts the first window dataframe into the list
				if rows_used >= t_window:
					windows.append(a_df[0:t_window].to_numpy())
				
				#Puts the remaining windows into the list
				for i in range(offset, rows_used, offset):
					windows.append( a_df[i:i+t_window].to_numpy() )	
			else:
				#This puts the first window dataframe into the list
				if rows_used >= t_window:
					loso_windows.append(a_df[0:t_window].to_numpy())
				
				#Puts the remaining windows into the list
				for i in range(offset, rows_used, offset):
					loso_windows.append( a_df[i:i+t_window].to_numpy() )	
		
		windows = np.array(windows)
		
		y_axis = df.columns.get_loc("activityID")
	
		#Delete columns we don't want
		cols_to_delete = []
		cols_to_delete.append(df.columns.get_loc("timestamp_s"))
		cols_to_delete.append(df.columns.get_loc("subject_id"))
		cols_to_delete.append(y_axis)
		
		y = windows[:, 0, y_axis]
		y = to_categorical(y)
		
		num_windows0 = windows.shape[0]
		x = windows
		
		if self.loso == False:
			x = np.delete(x, cols_to_delete, axis = 2)
			
			#Train size
			frac = self.train_percent
			train_size = int(x.shape[0]*frac)
			train_indices = list(random.sample(range(0, x.shape[0]), train_size))
			all_values = np.arange(0, num_windows0)
			test_indices = [ti for ti in all_values if ti not in train_indices]
			
			x_train = np.array([x[i,:,:] for i in train_indices])
			y_train = np.array([y[i,:] for i in train_indices])
			x_test = np.array([x[i,:,:] for i in test_indices])
			y_test = np.array([y[i,:] for i in test_indices])
		else:
			loso_windows = np.array(loso_windows)

			y_test = loso_windows[:, 0, y_axis]
			y_test = to_categorical(y_test)
			
			x_test = np.copy(loso_windows)
			x_test = np.delete(x_test, cols_to_delete, axis = 2)

			y_train = y
			x_train = x
			x_train = np.delete(x_train, cols_to_delete, axis = 2)
			
		#print("Loading PAMAP2 with Windows not implemented yet")
		return (x_train, y_train, x_test, y_test)
	
	def load_dataset_pamap2_nowindows(self):
		df = pd.read_csv("data/PAMAP2_Dataset/Protocol/ComboPlatter.csv")
			
		#Need to make activities go from 0-12 for the to_categorical function
		df.loc[df['activityID'] == 12, 'activityID'] = 8
		df.loc[df['activityID'] == 13, 'activityID'] = 9
		df.loc[df['activityID'] == 16, 'activityID'] = 10
		df.loc[df['activityID'] == 17, 'activityID'] = 11
		df.loc[df['activityID'] == 24, 'activityID'] = 12
		
		if self.loso == False:
			df = df.drop(["timestamp_s", "subject_id"], axis = 1)
			
			x_train = df.sample(frac = self.train_percent, random_state = 42)
			x_test = df.drop(x_train.index)
		else:
			#get all subject ID's
			df_subid = df['subject_id'].unique()

			rand_subject = -1
			while True:
				rand_subject = int(random.choice(df_subid))
				#This makes sure that the subject has all 13 activities
				if len((df[df["subject_id"] == rand_subject])['activityID'].unique()) == 13:
					break
			#rand_subject = int(random.choice(df_subid))
				
			x_train = df.loc[df['subject_id'] != rand_subject]
			x_test = df.loc[df['subject_id'] == rand_subject]
			
		y_train = x_train.pop('activityID')
		y_test = x_test.pop('activityID')
		
		#Makes them categorical, (one hot encoded over X columns)
		y_train = to_categorical(y_train, num_classes = 13)
		y_test = to_categorical(y_test, num_classes = 13)
	
		#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	
		#return x_train, y_train, x_test, y_test
		return x_train.to_numpy(), y_train, x_test.to_numpy(), y_test

#------------------------------------------------------------------------------
#START| LOADING FIREBUSTERS DATASET
	def load_dataset_fb_windows(self, t_window = 200, t_overlap = 0.25):
		df = pd.read_csv("data/Zenshin_Data/ComboPlatter.csv")
	
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

		#df_subid
		rand_subject = -1
		if self.loso == True:
			rand_subject = int(random.choice(df_subid))
	
		#Makes the windows
		loso_windows = []
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
			
			sid = int(a_df["subject_id"].iloc[0])
			if (self.loso == False) or (sid != rand_subject):
				#This puts the first window dataframe into the list
				if rows_used >= t_window:
					windows.append( a_df[0:t_window].to_numpy() )
				
				#Puts the remaining windows into the list
				for i in range(offset, rows_used, offset):
					windows.append( a_df[i:i+t_window].to_numpy() )
			else:
				#This puts the first window dataframe into the list
				if rows_used >= t_window:
					loso_windows.append(a_df[0:t_window].to_numpy())
				
				#Puts the remaining windows into the list
				for i in range(offset, rows_used, offset):
					loso_windows.append(a_df[i:i+t_window].to_numpy() )
		
		windows = np.array(windows)
		
		y_axis = df.columns.get_loc("exercise_id")
	
		#Delete columns we don't want
		cols_to_delete = []
		cols_to_delete.append(df.columns.get_loc("TimeStamp_s"))
		cols_to_delete.append(df.columns.get_loc("exercise_amt"))
		cols_to_delete.append(df.columns.get_loc("session_id"))
		cols_to_delete.append(df.columns.get_loc("subject_id"))
		cols_to_delete.append(y_axis)

		y = windows[:, 0, y_axis] - 1
		y = to_categorical(y)
		
		x = np.copy(windows)
		
		if self.loso == True:
			loso_windows = np.array(loso_windows)

			y_test = loso_windows[:, 0, y_axis] - 1
			y_test = to_categorical(y_test)
			
			x_test = np.copy(loso_windows)
			x_test = np.delete(x_test, cols_to_delete, axis = 2)

			y_train = y
			x_train = x
			x_train = np.delete(x_train, cols_to_delete, axis = 2)
		else:
			x = np.delete(x, cols_to_delete, axis = 2)

			#Train size
			frac = self.train_percent
			train_size = int(x.shape[0]*frac)
			train_indices = list(random.sample(range(0, x.shape[0]), train_size))
			all_values = np.arange(0, windows.shape[0])
			test_indices = [ti for ti in all_values if ti not in train_indices]
			
			x_train = np.array([x[i,:,:] for i in train_indices])
			y_train = np.array([y[i,:] for i in train_indices])
			x_test = np.array([x[i,:,:] for i in test_indices])
			y_test = np.array([y[i,:] for i in test_indices])
		
		#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
		return (x_train, y_train, x_test, y_test)

	def load_dataset_fb_windows_folds(self, t_window = 200, t_overlap = 0.25):
		df = pd.read_csv("data/Zenshin_Data/ComboPlatter.csv")
	
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

		df_subid.sort()
		#The validation subject
		rand_subject = df_subid[self.val_index]
		
		#Makes the windows
		loso_windows = []
		train_windows = [[] for _ in np.arange(max(df_subid)+1)]
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
			
			sid = int(a_df["subject_id"].iloc[0])
			if sid != rand_subject:
				#This puts the first window dataframe into the list
				if rows_used >= t_window:
					train_windows[sid-1].append( a_df[0:t_window].to_numpy() )
				
				#Puts the remaining windows into the list
				for i in range(offset, rows_used, offset):
					train_windows[sid-1].append( a_df[i:i+t_window].to_numpy() )
			else:
				#This puts the first window dataframe into the list
				if rows_used >= t_window:
					loso_windows.append(a_df[0:t_window].to_numpy())
				
				#Puts the remaining windows into the list
				for i in range(offset, rows_used, offset):
					loso_windows.append(a_df[i:i+t_window].to_numpy() )
		
		#Removes all the empty entries in train_windows
		tw = []
		for i in np.arange(len(train_windows)):
			if train_windows[i] != []:
				tw.append(train_windows[i])
		train_windows = tw

		for i in np.arange(len(train_windows)):
			train_windows[i] = np.array(train_windows[i])
#		windows = np.array(windows)
		
		y_axis = df.columns.get_loc("exercise_id")
	
		#Delete columns we don't want
		cols_to_delete = []
		cols_to_delete.append(df.columns.get_loc("TimeStamp_s"))
		cols_to_delete.append(df.columns.get_loc("exercise_amt"))
		cols_to_delete.append(df.columns.get_loc("session_id"))
		cols_to_delete.append(df.columns.get_loc("subject_id"))
		cols_to_delete.append(y_axis)

		all_y = []
		for i in np.arange(len(train_windows)):
			y = train_windows[i][:, 0, y_axis] - 1
			y = to_categorical(y)
			all_y.append(y)
		
		x = train_windows
		#x = np.copy(windows)
		
		loso_windows = np.array(loso_windows)

		y_test = loso_windows[:, 0, y_axis] - 1
		y_test = to_categorical(y_test)
		
		x_test = np.copy(loso_windows)
		x_test = np.delete(x_test, cols_to_delete, axis = 2)

		y_train = all_y
		x_train = x
		
		for i in np.arange(len(train_windows)):
			x_train[i] = np.delete(x_train[i], cols_to_delete, axis = 2)
		
		return (x_train, y_train, x_test, y_test)
	
	# load the dataset, returns train and test X and y elements
	def load_dataset_fb_nowindows(self):
		df = pd.read_csv("data/Zenshin_Data/ComboPlatter.csv")
		
		if self.loso == False:
			df = df.drop(["TimeStamp_s", "exercise_amt", "session_id", "subject_id"], axis = 1)
			
			x_train = df.sample(frac = self.train_percent, random_state = 0)
			x_test = df.drop(x_train.index)
		else:
			#get all subject ID's
			df_subid = df['subject_id'].unique()
			rand_subject = int(random.choice(df_subid))
				
			train = df.loc[df['subject_id'] != rand_subject]
			test = df.loc[df['subject_id'] == rand_subject]

			x_train = train.drop(["TimeStamp_s", "exercise_amt", "session_id", "subject_id"], axis = 1)
			x_test = test.drop(["TimeStamp_s", "exercise_amt", "session_id", "subject_id"], axis = 1)

		y_train = x_train.pop('exercise_id')
		y_test = x_test.pop('exercise_id')
	
		#Labels start at 1, so we will subtract 1 from the y's so they start at 0
		y_train = y_train-1
		y_test = y_test-1
	   	
		#Makes them categorical, (one hot encoded over 3 columns)
		y_train = to_categorical(y_train)
		y_test = to_categorical(y_test)
 
	  	#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	
		#return x_train, y_train, x_test, y_test
		return x_train.to_numpy(), y_train, x_test.to_numpy(), y_test

	# load the dataset, returns train and test X and y elements
	def load_dataset_fb_nowindows_folds(self):
		df = pd.read_csv("data/Zenshin_Data/ComboPlatter.csv")
		
		#get all subject ID's
		df_subid = df['subject_id'].unique()
		#rand_subject = self.val_index+1
		#rand_subject = int(random.choice(df_subid))

		df_subid.sort()
		#The validation subject
		rand_subject = df_subid[self.val_index]
		
		#Has an array of dataframes
		x_train = []
		y_train = []
		for subid in df_subid:
			if subid != rand_subject:
				df_train = df.loc[df['subject_id'] == subid]
				xtrain = df_train.drop(["TimeStamp_s", "exercise_amt", "session_id", "subject_id"], axis = 1)
				ytrain = xtrain.pop('exercise_id')
				ytrain = ytrain-1
				ytrain = to_categorical(ytrain)
				x_train.append(xtrain.to_numpy())
				y_train.append(ytrain)

		#In this instance test is the LOSO validation subject
		test = df.loc[df['subject_id'] == rand_subject]
		x_test = test.drop(["TimeStamp_s", "exercise_amt", "session_id", "subject_id"], axis = 1)
		y_test = x_test.pop('exercise_id')
	   	#Labels start at 1, so we will subtract 1 from the y's so they start at 0
		y_test = y_test-1
	   	#Makes them categorical, (one hot encoded over 3 columns)
		y_test = to_categorical(y_test)
 
		return x_train, y_train, x_test.to_numpy(), y_test

	#When we're doing K-Fold cross validation we need to break that training
	#set up into folds.
	#foldNum indicates which fold is the test
	#foldSize indicates how large a fold is
	def GetTrainTestFromFolds(self, testIndices):
		xtrain = []
		xtest = []
		ytrain = []
		ytest = []
		
		for i in np.arange(len(self.x_train)):
			if i not in testIndices:
				#This is the training
				xtrain.append(self.x_train[i])
				ytrain.append(self.y_train[i])
			else:
				#This is the test
				xtest.append(self.x_train[i])
				ytest.append(self.y_train[i])
		
		xtrain = np.concatenate(xtrain)
		ytrain = np.concatenate(ytrain)

		xtest = np.concatenate(xtest)
		ytest = np.concatenate(ytest)

		#For ConvLSTM2D we gotta reshape the data		
		if self.clstm_params != {}:
			n_features = xtrain.shape[2]
			# reshape into subsequences (samples, time steps, rows, cols, channels)
			n_steps = self.clstm_params["n_steps"][0]
			if (self.window_size%n_steps) != 0:
				n_steps = 5
			n_length = int(xtrain.shape[1]/n_steps)
			xtrain = xtrain.reshape((xtrain.shape[0], n_steps, 1, n_length, n_features))
			xtest = xtest.reshape((xtest.shape[0], n_steps, 1, n_length, n_features))
		
		print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
		return xtrain, ytrain, xtest, ytest

#ld = Load_Data('firebusters', w_size = 200)
#ld = Load_Data('pamap2', w_size = 200)
#ld = Load_Data('har')
'''
from analysis.zg_layer_generator_01 import Layer_Generator

lay_gen = Layer_Generator()
#if model_structures_type == "ConvLSTM2D":
clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
#clstm_params = {}


#folds_heldout is the id of the dude left out
val_index = 0
	
data_params = {'dataset' : 'firebusters',
				'train_p' : 0.8,
				'w_size' : 200,
				'o_percent' : 0,
				'LOSO' : True,
				'clstm_params' : clstm_params
#				'valIndex' : val_index
				}
dataset = Load_Data(**data_params)

#train = dataset.x_train
#print(dataset.x_test.shape)

testIndices = [0,3,4]
x_train, y_train, x_test, y_test = dataset.GetTrainTestFromFolds(testIndices)
x_val = dataset.x_test
y_val = dataset.y_test

print(x_val.shape, y_val.shape)
'''