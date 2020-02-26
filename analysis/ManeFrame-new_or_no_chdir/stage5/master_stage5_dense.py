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
import os
import statistics
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
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

def Get_Accuracy(y_true, y_pred):
	correct = 0

	for i in np.arange(len(y_true)):
		if y_true[i][y_pred[i]] == 1:
			correct += 1.0
	
	return correct/float(len(y_true))

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

print(acc["Conv1D"][0])

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

#==============================================================================

#Stage 5 function
def Stage_Five(best_setups, best_idx, model_structures_type):
	#Putting the structures into this ordering
	model_structures = []
	hyp_str = []
	for setup in best_setups:
		model_structures.append(setup[0][0])
		hyp_str.append(setup[1])
	
	results_here = []
	for i in np.arange(len(model_structures)):
		with open("data/step5_hyp/" + model_structures_type + "_ModelStr_Hyp.pkl", "wb") as fp:   #Pickling
			pickle.dump(hyp_str[i], fp)
			
		window_size = best_setups[i][0][5][0]
		overlap_per = float(best_setups[i][0][5][1])/float(100.0)
		batch_size = best_setups[i][0][5][2]
	
		#List of model accuracies for the 28 validations
		val_acc = []
		test_acc = []
		val_indices = []
		test_indices = []
		for val_index in np.arange(28):
			print("SUBJECT:", val_index)
			#ConvLSTM2D has extra stuff, so if this is that it gets the parameters
			lay_gen = Layer_Generator()
			clstm_params = {}
			if model_structures_type == "ConvLSTM2D":
				clstm_params = lay_gen.Generate_Layer_Parameters()["ConvLSTM2D"]
			
			#print(clstm_params)
			
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
			test_set_size = 3
			testIndices = random.sample(list(np.arange(27)), test_set_size)
			#Based on the test indices, it makes the training/test sets
			x_train, y_train, x_test, y_test = dataset.GetTrainTestFromFolds(testIndices)
			x_val = dataset.x_test
			y_val = dataset.y_test
			
			#Because we send it the dataset, we make a fake dataset class that will
			#contain the data so that it can be referenced
			fake_dataset = FakeDataset(x_train, x_test, y_train, y_test, window_size)
			mt = Model_Tuning([model_structures[i]],
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
	
			accuracy = Get_Accuracy(y_val, result_val)
			val_acc.append(accuracy)

			#Because of the way that the indices are made, when the val dude
			#is removed, the id list works like a stack. So this makes it so that
			#the values are able to be compared during our analysis.
			for i in np.arange(len(testIndices)):
				if testIndices[i] >= val_index:
					testIndices[i] = testIndices[i] + 1
			
			test_indices.append(testIndices)
			val_indices.append(val_index)
			test_acc.append(max(result_train.history["val_accuracy"][-8:]))
			#END FOR LOOP
		
		data_params = best_setups[i][0][5]
		first_idx = [model_structures[i], best_idx[i], test_acc, test_indices, val_acc, val_indices, data_params]
		res_loop = [first_idx, hyp_str[i]]
		#Appends the results from this model/validation set
		results_here.append(res_loop)
		#Delete the pkl file
		os.remove("data/step5_hyp/" + model_structures_type + "_ModelStr_Hyp.pkl") 
		#END FOR LOOP

	#SAVE THE RESULTS TO A FILE
	with open("data/step5_res/" + model_structures_type + "_results.pkl", "wb") as fp:   #Pickling
		pickle.dump(results_here, fp)

	return

#==============================================================================

#Chooses the ones with the highest accuracy based on conditions
#ConvLSTM2D          65
#Conv1D              34
#df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.03) & (df["val_accuracy"] >= 0.9) ]
#BidirectionalGRU    48
#BidirectionalLSTM   46
#df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.05) & (df["val_accuracy"] >= 0.9) ]
#LSTM                38
#df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.08) & (df["val_accuracy"] >= 0.9) ]
#GRU                 37
#df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.1) & (df["val_accuracy"] >= 0.8) ]
#Dense               34
df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.3) & (df["val_accuracy"] >= 0.6) ]

#print( df_best[["key"]].apply(pd.value_counts) )

#Model Category
#["BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D", "Dense", "GRU", "LSTM"]
model_structures_type = "Dense"

#Select the model category
df_best = df_best[df_best["key"] == model_structures_type]

#Gets the best indices (indices are from the original pkl file)
best_idx = list(df_best["index"])

#Randomly chooses 20 of them to run
best_idx = random.sample(best_idx, 20)

#Pull out those best models
best_setups = acc[model_structures_type]
best_setups = [best_setups[i] for i in best_idx ]

Stage_Five(best_setups, best_idx, model_structures_type)





