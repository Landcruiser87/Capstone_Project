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

import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data

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

def model_nn_01(x_train, y_train, x_test, y_test, x_val, y_val, epochs = 10, batch_size = 100):
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
	results_val = model.predict_classes(x_val)

	return results.history["val_accuracy"][-1]


val_index = 0

data_params = {'dataset' : 'firebusters',
               'train_p' : 0.8,
               'w_size' : 200,
               'o_percent' : 0, 
			   	'LOSO' : True,
				# 'clstm_params' : clstm_params,
				'valIndex' : val_index
               }
dataset = Load_Data(**data_params)

model_params = {'epochs' : 10,
                'batch_size' : 512
                }

				
testIndices = [0,3,4]
x_train, y_train, x_test, y_test = dataset.GetTrainTestFromFolds(testIndices)
x_val = dataset.x_test
y_val = dataset.y_test

print(x_val.shape, y_val.shape)

run_experiment(dataset.x_train, dataset.y_train, 
				dataset.x_test, dataset.y_test,
                dataset.x_val, dataset.y_val, model_nn_01, model_params, 1)



# def moving_average(data_set, periods=3):
# 	weights = np.ones(periods) / periods
# 	return np.convolve(data_set, weights, mode='valid')

# def ma(window, n):
# 	return np.vstack([moving_average(window[:,i], n) for i in range(window.shape[-1])]).T

# def ma_batch(batch, n):
# 	return np.dstack([ma(batch[i,:,:], n) for i in range(batch.shape[0])])
#------------------------------------------------------------------------------
#START| MAIN

# batch = 512
# epochs = 100
# inputs = Input(....)
# x = Dense(128)(input)
# x = LSTM(32....)(x)
# out = Dense(softmax)(x)
# 10 fold CV
        # 3 subjects or 1 repetition
        # v_subjects = np.array(np.unique(self.subject)[3:6])
        # v_reps = np.array(np.unique(self.rep)[3::2])
        # # accepts a tuple (is_validation, by_subject)
        # case_dict = {
        #         (False, False):np.where(np.isin(self.rep, v_reps, invert=True)),
        #     # last rep is for testing
        #         (True, False):np.where(np.isin(self.rep, v_reps[:-1])),
        #         (False, True):np.where(np.isin(self.subject, v_subjects, invert=True)),
        #     # last subject is for testing
        #         (True, True):np.where(np.isin(self.subject, v_subjects[:-1]))
        #         }
        # # return the indices we want
        # case = case_dict[(validation, by_subject)]
        # # construnct test data
        # if validation:
        #     if not by_subject:
        #         test_id = np.where(np.isin(self.rep, v_reps[-1])),
        #         self.test_data = (self.emg[test_id][0,:], self.labels[test_id][0,:])
        #     else:
        #         test_id = np.where(np.isin(self.subject, v_subjects[-1]))
        #         self.test_data = (self.emg[test_id], self.labels[test_id])
        # self._indexer(case)
    # def _indexer(self, id):
    #     self.emg = self.emg[id]
    #     self.rep = self.rep[id]
    #     self.labels=self.labels[id]
    #     self.subject=self.subject[id]

#dataset, train_p = 0.8, w_size = 0, o_percent = 0.25):
# lay_gen = Layer_Generator()
# #if model_structures_type == "ConvLSTM2D":
# clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
