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
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import numpy as np
import pandas as pd
import pickle
import random
from random import sample 
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.finalcountdown_stage4 import Final_Accuracy
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator
from analysis.zg_model_tuning_01 import Model_Tuning
from analysis.ah_Model_Info import Model_Info

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#Returns the percentage of layers that are of the given type
def Percent_Layer_Type(layers, s):
	count = len([i for i in layers if i.startswith(s)])
	return float(count)/float(len(layers))

#Gets the average "size" of a layer
def Avg_Num_Nodes(hyp):
	val = 0
	count = 0
	for key in hyp:
		if "units" in hyp[key]:
			count += 1
			val += hyp[key]["units"][0]

	if count == 0:
		return 0

	return int(float(val)/float(count))

class FakeTuner:
	def Choice(self, name = "", ls = []):
		# hp.Choice("model_structures_index", self.model_structures_index)
		return random.choice(ls)

class FakeDataset:
	def __init__(self, xtrain, xtest, ytrain, ytest, ws):
		self.x_train = xtrain
		self.x_test = xtest
		self.y_train = ytrain
		self.y_test = ytest
		self.window_size = ws
		return
#==============================================================================

#Load in the accuracy files
fa = Final_Accuracy()
fa.Help()
acc = fa.Load_The_File()

colnames = ["key", "index", "val_accuracy", "train_accuracy", "depth", "optimizer"]
#key, acc, val_acc, layer depth, optimizer
rows = []
for key in acc:
	i = 0
	for model in acc[key]:
		row = []
		row.append(key)						#key
		row.append(i)						#index of where it was in orig
		row.append(*model[0][3])			#validation accuracy
		row.append(*model[0][1])			#training accuracy
		row.append(len(model[0][0]))		#depth
		row.append(model[1]["optimizer"])	#optimizer
		row.append(Percent_Layer_Type(model[0][0], "Dropout"))
		row.append(Percent_Layer_Type(model[0][0], "GRU"))
		row.append(Percent_Layer_Type(model[0][0], "LSTM"))
		row.append(Percent_Layer_Type(model[0][0], "BidirectionalGRU"))
		row.append(Percent_Layer_Type(model[0][0], "BidirectionalLSTM"))
		row.append(Percent_Layer_Type(model[0][0], "Dense"))
		row.append(Percent_Layer_Type(model[0][0], "Conv1D"))
		row.append(Percent_Layer_Type(model[0][0], "ConvLSTM2D"))
		row.append(Percent_Layer_Type(model[0][0], "MaxPooling"))
		row.append(model[0][5][0])
		row.append(model[0][5][1])
		row.append(model[0][5][2])
		row.append(Avg_Num_Nodes(model[1]))
		rows.append(row)
		i += 1

#Makes the rows into a dataframe
df = pd.DataFrame(rows)
#Sets the names of the columns
colnames.append("lay_%dropout")
colnames.append("lay_%gru")
colnames.append("lay_%lstm")
colnames.append("lay_%bgru")
colnames.append("lay_%blstm")
colnames.append("lay_%dense")
colnames.append("lay_%conv1d")
colnames.append("lay_%cl2d")
colnames.append("lay_%pool")
colnames.append("window_size")
colnames.append("overlap%")
colnames.append("batch_size")
colnames.append("avg_num_nodes")
df.columns = colnames

#------------------------------------------------------------------------------

#Chooses the ones with the highest accuracy based on conditions
df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.1) & (df["val_accuracy"] >= 0.9) ]

#Model Category
model_structures_type = "Conv1D"

#Select the model category
df_best = df_best[df_best["key"] == model_structures_type]

#Gets the best indices (indices are from the original pkl file)
best_idx = list(df_best["index"])

#Pull out those best models
best_setups = acc[model_structures_type]
best_setups = [best_setups[i] for i in best_idx ]

#Putting the structures into this ordering
model_structures = []
hyp_str = []
for setup in best_setups:
	model_structures.append(setup[0][0])
	hyp_str.append(setup[0][1])

#def Data_Hyperparameter_Tuning(model_structures_type, model_structures, hyp_str):

for i in np.arange(len(model_structures)):
	with open("data/step5_hyp/" + model_structures_type + "_ModelStr_Hyp.pkl", "wb") as fp:   #Pickling
		pickle.dump(hyp_str, fp)
		
	window_size = best_setups[i][0][5][0]
	overlap_per = float(best_setups[i][0][5][1])/float(100.0)
	batch_size = best_setups[i][0][5][2]

	#List of model accuracies for the 28 validations
	val_acc = []
	train_acc = []
	test_acc = []
	val_index = []
	test_indices = []
	for val_index in np.arange(28):
		#ConvLSTM2D has extra stuff, so if this is that it gets the parameters
		lay_gen = Layer_Generator()
		if model_structures_type == "ConvLSTM2D":
			clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
		clstm_params = {}

		print("TODO: LOAD IN DATA PARAMETERS")
		#Loading in the dataset
		data_params = {'dataset' : 'firebusters',
						'train_p' : 0.8,
						'w_size' : window_size,
						'o_percent' : overlap_per,
						'LOSO' : True,
						'clstm_params' : clstm_params,
						'valIndex' : val_index
						}
		dataset = Load_Data(**data_params)

		#The indices of the test set (0-27), the other ones are made into the train
		testIndices = sample(np.arange(27), 3)
		#Based on the test indices, it makes the training/test sets
		x_train, y_train, x_test, y_test = dataset.GetTrainTestFromFolds(testIndices)
		x_val = dataset.x_test
		y_val = dataset.y_test
		
		fake_dataset = FakeDataset(x_train, x_test, y_train, y_test, window_size)
		#TODO: because we send it the dataset, we need to change how we do this
		#and make it so that this will work
		mt = Model_Tuning(model_structures,
							fake_dataset,
							m_tuning = "val_" + model_structures_type,
							parent_fldr = "",
							fldr_name = "",
							fldr_sffx = "")

		hp = FakeTuner()
		model = mt.The_Model(hp)
		callbacks = [EarlyStopping(monitor='val_accuracy', mode='max', patience = 8, restore_best_weights=True)]
		result_train = model.fit(x_train, y_train, validation_data=(x_test, y_test),
							  epochs = 60, batch_size = batch_size, callbacks=callbacks)
		result_val = model.predict_classes(x_val)

		print("HOW TO GET THE ACCURACY") #https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
		print("Save the val/train/test accuracy, Save indices of test set, save val index")
	print("Save the lists of stuff")
	#Delete the pkl file
	os.remove("data/step5_hyp/" + model_structures_type + "_ModelStr_Hyp.pkl") 
	



#	return results.history["val_accuracy"][-1]
	
		#mt.Tune_Models(epochs = 60, batch_size = params[2], MAX_TRIALS = 20)
	
#LOOP over all models
	#Save the parameters to the data folder

	#LOOP 28 times for the LOSO dude
		#Load the data
		#Select a random test set
		#Return the model with model_tuning.The_Model(self, hp)
		#Run the model and save the weights

		#Run the model with weights on LOSO dude
		#record val acc, train acc, test acc, test idx's



#modify the layer generator so that it has a val category in the 
	#Generate_Layer_Parameters(self) function

















