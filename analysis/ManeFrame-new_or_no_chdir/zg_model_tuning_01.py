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
import numpy as np

import os
#os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.zg_Load_Data import Load_Data
from analysis.zg_layer_generator_01 import Layer_Generator

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#START MODEL CONSTRUCTION AND TUNING

class Model_Tuning:
	
	def __init__(self, model_str, data, m_tuning = "all", parent_fldr = "test_dir", fldr_name = "TYPE_", fldr_sffx = ""):
		self.model_tuning = m_tuning
		self.model_structures = model_str
		self.folder_suffix = fldr_sffx
		self.parent_folder = parent_fldr
		self.folder_name = fldr_name
		msi = list(np.arange(len(model_str)))
		self.model_structures_index = []
		self.dataset = data
		self.good_layer_index = -1
		for i in msi:
			self.model_structures_index.append(int(i))
		return
	
	#This function returns the model. For the keras-tuner this is the function
	#that it runs
	def The_Model(self, hp): #layer_parameters, model_structures):
		model = Sequential()
	
		lay_gen = Layer_Generator(self.model_tuning)
		all_layer_params = lay_gen.Generate_Layer_Parameters()
		bias_init = None
		model_index = hp.Choice("model_structures_index", self.model_structures_index)
		chosen_model = self.model_structures[model_index]
		
		#print(chosen_model)
		
		#Find the index of the layer setup in the case of this being for data tuning
		if self.model_tuning.startswith("data"):
			for i in np.arange(len(all_layer_params)):
				if all_layer_params[i][0] == chosen_model:
					self.good_layer_index = i     #Dont remember what this does
					all_layer_params = all_layer_params[i][1]
					break
		if self.model_tuning.startswith("val"):
			self.good_layer_index = 1

		#print(all_layer_params)

		#Goes through each layer
		for layer_index in np.arange(len(chosen_model)):
			#print(chosen_model[layer_index])
			if chosen_model[layer_index] == "GRU":
				bias_init, layers = self.Add_GRU_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "LSTM":
				bias_init, layers = self.Add_LSTM_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "Dense":
				bias_init, layers = self.Add_Dense_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "BidirectionalLSTM":
				bias_init, layers = self.Add_BidirectionalLSTM_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "BidirectionalGRU":
				bias_init, layers = self.Add_BidirectionalGRU_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "Conv1D":
				bias_init, layers = self.Add_Conv1D_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "ConvLSTM2D":
				bias_init, layers = self.Add_ConvLSTM2D_Layer(hp, bias_init, chosen_model, layer_index, all_layer_params)
				for l in layers:
					model.add(l)
			elif chosen_model[layer_index] == "Dropout":
				model.add(self.Add_Dropout_Layer(hp, layer_index, all_layer_params))
			elif chosen_model[layer_index].startswith("MaxPooling"): #Fix for 1D, 2D, 3D
				model.add(self.Add_MaxPooling1D_Layer(hp, layer_index, chosen_model, all_layer_params))
			elif chosen_model[layer_index] == "Flatten":
				model.add(self.Add_Flatten_Layer())
		
		model.add(Dense(3, activation='softmax'))
		
		optimizer = "adam"
		#For the data tuning stage (stage 4)
		if self.model_tuning.startswith("data"):
			optimizer = hp.Choice("optimizer", ["adam", "RMSprop"])
		elif self.model_tuning.startswith("val"):
			optimizer = hp.Choice("optimizer", [all_layer_params["optimizer"]])
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		
		#print(model.get_config())
		#print(model.summary())
		#print("The_Model is not completed")
		return model
	
	def Add_GRU_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "GRU_" + str(layer_index) + "_"
		
		#GRU, Flatten, GRU
		#GRU, Dense,   GRU
		#GRU,  GRU
		#0,	 1,	  2
	
		#GRU_0_Activation	  A, B, C
		#GRU_1_Activation
		#GRU_2_Activation
		
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["GRU"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["GRU_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		#Random choice for these two parameters
		units = hp.Choice(name_prefix + "units", layer_parameters["units"])
		dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])
	
		#all bias_initializer in the model should be the same
		if bias_init == None:
			print("Set the bias_initializer from the random thingy")
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
		else:
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
		
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_initializer == "Zeros":
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
		elif bias_initializer == "glorot_normal":
			activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
		elif bias_initializer == "RandomNormal":
			activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
	
		#if a layer after this is a RNN type or a Dropout-RNN type then
		#return_sequences can be either true or false, otherwise it is false
		if len(all_layers) > (layer_index + 1):
			rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
			if all_layers[layer_index + 1] in rnn_types:	#GRU-RNN type
				return_sequences = True #hp.Choice(name_prefix + "return_sequences_t", [True]) # layer_parameters["return_sequences"])
			else:
				if all_layers[layer_index + 1] == "Dropout":
					if len(all_layers) > (layer_index + 2):
						if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
							return_sequences = True #hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
						else:
							return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
					else:
						return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
				else:
					return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
		else:
			return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])
	
		#If this is the first layer in the model, this has to have the input shape
		#fed into it
		if layer_index == 0:
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = GRU(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]))
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = GRU(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]))
				these_layers.append(layer)
		else: 
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = GRU(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer)
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = GRU(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer)
				these_layers.append(layer)
				
		return (bias_init, these_layers)
	
	def Add_LSTM_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "LSTM_" + str(layer_index) + "_"
		
		#We do this here just because we can
		#layer_parameters = all_layer_params["LSTM"]
		
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["LSTM"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["LSTM_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		#Random choice for these two parameters
		units = hp.Choice(name_prefix + "units", layer_parameters["units"])
		dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])
	
		#all bias_initializer in the model should be the same
		if bias_init == None:
			print("Set the bias_initializer from the random thingy")
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
		else:
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
		
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_initializer == "Zeros":
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
		elif bias_initializer == "glorot_normal":
			activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
		elif bias_initializer == "RandomNormal":
			activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
	
		#if a layer after this is a RNN type or a Dropout-RNN type then
		#return_sequences can be either true or false, otherwise it is false
		if len(all_layers) > (layer_index + 1):
			rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
			if all_layers[layer_index + 1] in rnn_types:	#GRU-RNN type
				return_sequences = True #hp.Choice(name_prefix + "return_sequences_t", [True])# layer_parameters["return_sequences"])
			else:
				if all_layers[layer_index + 1] == "Dropout":
					if len(all_layers) > (layer_index + 2):
						if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
							return_sequences = True #hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
						else:
							return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
					else:
						return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
				else:
					return_sequences = False # hp.Choice(name_prefix + "return_sequences_f", [False])
		else:
			return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])
		
		#If this is the first layer in the model, this has to have the input shape
		#fed into it
		if layer_index == 0:
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = LSTM(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]))
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = LSTM(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]))
				these_layers.append(layer)
		else: 
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = LSTM(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer)
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = LSTM(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer)
				these_layers.append(layer)
				
		return (bias_init, these_layers)
	
	#Adds the Dense layer and hyperparameters by condition
	def Add_Dense_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "Dense_" + str(layer_index) + "_"
		
		#We do this here just because we can
		#layer_parameters = all_layer_params["Dense"]
				
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["Dense"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["Dense_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		#Random choice for these two parameters
		units = hp.Choice(name_prefix + "units", layer_parameters["units"])
		#dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])
	
		#all bias_initializer in the model should be the same
		if bias_init == None:
			print("Set the bias_initializer from the random thingy")
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
		else:
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
		
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_initializer == "Zeros":
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
		elif bias_initializer == "glorot_normal":
			activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
		elif bias_initializer == "RandomNormal":
			activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
	
		#If this is the first layer in the model, this has to have the input shape
		#fed into it
		if layer_index == 0:
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = Dense(units = units,
							bias_initializer = bias_initializer,
							input_dim = self.dataset.x_train.shape[1])
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = Dense(units = units, activation = activation,
							bias_initializer = bias_initializer,
							input_dim = self.dataset.x_train.shape[1])
				these_layers.append(layer)
		else: 
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = Dense(units = units,
							bias_initializer = bias_initializer)
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = Dense(units = units, activation = activation,
							bias_initializer = bias_initializer)
				these_layers.append(layer)
				
		return (bias_init, these_layers)
	
	def Add_BidirectionalLSTM_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "BidirectionalLSTM_" + str(layer_index) + "_"
		
		#We do this here just because we can
		#layer_parameters = all_layer_params["LSTM"]
				
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["LSTM"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["BidirectionalLSTM_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		#Random choice for these two parameters
		units = hp.Choice(name_prefix + "units", layer_parameters["units"])
		dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])
	
		#all bias_initializer in the model should be the same
		if bias_init == None:
			print("Set the bias_initializer from the random thingy")
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
		else:
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
		
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_initializer == "Zeros":
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
		elif bias_initializer == "glorot_normal":
			activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
		elif bias_initializer == "RandomNormal":
			activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
	
		#if a layer after this is a RNN type or a Dropout-RNN type then
		#return_sequences can be either true or false, otherwise it is false
		if len(all_layers) > (layer_index + 1):
			rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
			if all_layers[layer_index + 1] in rnn_types:	#GRU-RNN type
				return_sequences = True #hp.Choice(name_prefix + "return_sequences_t", [True])# layer_parameters["return_sequences"])
			else:
				if all_layers[layer_index + 1] == "Dropout":
					if len(all_layers) > (layer_index + 2):
						if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
							return_sequences = True #hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
						else:
							return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
					else:
						return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
				else:
					return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
		else:
			return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])
		
		#If this is the first layer in the model, this has to have the input shape
		#fed into it
		if layer_index == 0:
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = Bidirectional(LSTM(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2])))
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = Bidirectional(LSTM(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2])))
				these_layers.append(layer)
		else: 
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = Bidirectional(LSTM(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer))
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = Bidirectional(LSTM(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer))
				these_layers.append(layer)
				
		return (bias_init, these_layers)
	
	def Add_BidirectionalGRU_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "BidirectionalGRU_" + str(layer_index) + "_"
		
		#We do this here just because we can
		#layer_parameters = all_layer_params["GRU"]
		
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["GRU"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["BidirectionalGRU_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		#Random choice for these two parameters
		units = hp.Choice(name_prefix + "units", layer_parameters["units"])
		dropout = hp.Choice(name_prefix + "dropout", layer_parameters["dropout"])
	
		#all bias_initializer in the model should be the same
		if bias_init == None:
			print("Set the bias_initializer from the random thingy")
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", layer_parameters["bias_initializer"])
		else:
			bias_initializer = hp.Choice(name_prefix + "bias_initializer", [bias_init])
		
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_initializer == "Zeros":
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
		elif bias_initializer == "glorot_normal":
			activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
		elif bias_initializer == "RandomNormal":
			activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
	
		#if a layer after this is a RNN type or a Dropout-RNN type then
		#return_sequences can be either true or false, otherwise it is false
		if len(all_layers) > (layer_index + 1):
			rnn_types = ["GRU", "LSTM", "BidirectionalLSTM", "BidirectionalGRU"]
			if all_layers[layer_index + 1] in rnn_types:	#GRU-RNN type
				return_sequences = True #hp.Choice(name_prefix + "return_sequences_t", [True])# layer_parameters["return_sequences"])
			else:
				if all_layers[layer_index + 1] == "Dropout":
					if len(all_layers) > (layer_index + 2):
						if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
							return_sequences = True #hp.Choice(name_prefix + "return_sequences", [True])#layer_parameters["return_sequences"])
						else:
							return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
					else:
						return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
				else:
					return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])
		else:
			return_sequences = False #hp.Choice(name_prefix + "return_sequences_f", [False])# layer_parameters["return_sequences"])
		
		#If this is the first layer in the model, this has to have the input shape
		#fed into it
		if layer_index == 0:
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = Bidirectional(GRU(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2])))
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = Bidirectional(GRU(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer,
							input_shape = (self.dataset.x_train.shape[1], self.dataset.x_train.shape[2])))
				these_layers.append(layer)
		else: 
			#leakyReLU is a pain because it acts like it is its own layer
			if activation == "LeakyReLU":
				layer = Bidirectional(GRU(units = units, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer))
				these_layers.append(layer)
				these_layers.append(LeakyReLU())
			else:
				layer = Bidirectional(GRU(units = units, activation = activation, dropout = dropout,
							return_sequences = return_sequences,
							bias_initializer = bias_initializer))
				these_layers.append(layer)
				
		return (bias_init, these_layers)
	
	def Add_Conv1D_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		print(all_layer_params)
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "Conv1D_" + str(layer_index) + "_"
		
		#We do this here just because we can
		#layer_parameters = all_layer_params["Conv1D"]
				
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["Conv1D"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["Conv1D_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		#all bias_initializer in the model should be the same
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_init == None:
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
			if activation == "relu":
				bias_initializer = hp.Choice(name_prefix + "bias_init", ["Zeros", "RandomNormal"])
			elif activation == "tanh":
				bias_initializer = hp.Choice(name_prefix + "bias_init", ["glorot_normal", "Zeros"])
			elif activation == "LeakyReLU":
				bias_initializer = hp.Choice(name_prefix + "bias_init", ["Zeros", "RandomNormal"])
			else:
				print("Add_Conv1D_Layer, bias_init SOMETHING WENT WRONG")
		else:
			if bias_initializer == "Zeros":
				activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
			elif bias_initializer == "glorot_normal":
				activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
			elif bias_initializer == "RandomNormal":
				activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
			else:
				print("Add_Conv1D_Layer, bias_init SOMETHING WENT WRONG")
			
		#Convert filter percent to a int by multiplying it times the window_size
		filters = int(self.dataset.window_size*hp.Choice(name_prefix + "filters", layer_parameters["filters"]))
		#print("FILTERS:", filters)
		
		if filters < 1:
			filters = 1
		
		#TODO: kernel_size is window_size/filter (for our purposes)
		#kernel_size = filters
		kernel_size = int(filters*0.1)
		
		if kernel_size < 2:
			kernel_size = 2
		
		#If this is the first layer so stuffs
		if activation != "LeakyReLU":
			if layer_index == 0:
				n_timesteps, n_features = self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]
				layer = Conv1D(filters=filters, kernel_size=kernel_size,
										activation=activation,
										input_shape=(n_timesteps, n_features), padding="same")
				these_layers.append(layer)
			else:
				layer = Conv1D(filters=filters, kernel_size=kernel_size,
										activation=activation, padding="same")
				these_layers.append(layer)
		else:
			if layer_index == 0:
				n_timesteps, n_features = self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]
				layer = Conv1D(filters=filters, kernel_size=kernel_size,
										input_shape=(n_timesteps, n_features), padding="same")
				these_layers.append(layer)
			else:
				layer = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")
				these_layers.append(layer)
			these_layers.append(LeakyReLU())
	 
		return (bias_init, these_layers)
	
	def Add_ConvLSTM2D_Layer(self, hp, bias_init, all_layers, layer_index, all_layer_params):
		#Because of leakyReLU, we technically return multiple layers
		these_layers = []
		
		#The importance of this depends on how keras-tuner stores the parameters,
		#im assuming that it stores them by like a dictionary.
		name_prefix = "ConvLSTM2D_" + str(layer_index) + "_"
		
		#We do this here just because we can
		#layer_parameters = all_layer_params["ConvLSTM2D"]
				
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["ConvLSTM2D"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["ConvLSTM2D_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
	
		#all bias_initializer in the model should be the same
		#activation function can be different across layers
		#when activation is relu, use RandomNormal or Zero
		#when activation is leakyRelu, can use RandomNormal or Zero
		#when activation is tanh, use glorot_normal (Xavier) or Zero
		if bias_init == None:
			activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
			if activation == "relu":
				bias_initializer = hp.Choice(name_prefix + "bias_init", ["Zeros", "RandomNormal"])
			elif activation == "tanh":
				bias_initializer = hp.Choice(name_prefix + "bias_init", ["glorot_normal", "Zeros"])
			elif activation == "LeakyReLU":
				bias_initializer = hp.Choice(name_prefix + "bias_init", ["Zeros", "RandomNormal"])
			else:
				print("Add_ConvLSTM2D_Layer, bias_init SOMETHING WENT WRONG")
		else:
			if bias_initializer == "Zeros":
				activation = hp.Choice(name_prefix + "activation_zeros", layer_parameters["activation"])
			elif bias_initializer == "glorot_normal":
				activation = hp.Choice(name_prefix + "activation_glorot", ["tanh"])
			elif bias_initializer == "RandomNormal":
				activation = hp.Choice(name_prefix + "activation_randomnormal", ["relu", "LeakyReLU"])
			else:
				print("Add_ConvLSTM2D_Layer, bias_init SOMETHING WENT WRONG")
		
		#Convert filter percent to a int by multiplying it times the window_size
		filters = int(self.dataset.window_size*hp.Choice(name_prefix + "filters", layer_parameters["filters"]))
		
		if filters < 1:
			filters = 1
		
		#TODO: kernel_size is window_size/filter (for our purposes)
		#kernel_size = filters
		kernel_size = int(filters*0.1)
		
		if kernel_size < 2:
			kernel_size = 2
	
		#	n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
		  	# reshape into subsequences (samples, time steps, rows, cols, channels)
		#	n_steps = 4
		#	n_length = int(x_train.shape[1]/n_steps)
		#	x_train = x_train.reshape((x_train.shape[0], n_steps, 1, n_length, n_features))
		#	x_test = x_test.reshape((x_test.shape[0], n_steps, 1, n_length, n_features))
		#	ConvLSTM2D(filters=64,
		#			kernel_size=(1,3),
		#			activation='relu',
		#			input_shape=(n_steps, 1, n_length, n_features)))
		
		#if a layer after this is a RNN type or a Dropout-RNN type then
		#return_sequences can be either true or false, otherwise it is false
		if len(all_layers) > (layer_index + 1):
			rnn_types = ["ConvLSTM2D"]
			if all_layers[layer_index + 1] in rnn_types:	#GRU-RNN type
				return_sequences = True
			else:
				if all_layers[layer_index + 1] == "Dropout" or all_layers[layer_index + 1].startswith("MaxPooling"):
					if len(all_layers) > (layer_index + 2):
						if all_layers[layer_index + 2] in rnn_types: #GRU-Dropout-RNN type
							return_sequences = True
						else:
							if all_layers[layer_index + 2] == "Dropout" or all_layers[layer_index + 2].startswith("MaxPooling"):
								if len(all_layers) > (layer_index + 3):
									if all_layers[layer_index + 3] in rnn_types: #GRU-Dropout-MaxPool-RNN type
										return_sequences = True
									else:
										return_sequences = False
								else:
									return_sequences = False
							else:
								return_sequences = False
					else:
						return_sequences = False
				else:
					return_sequences = False
		else:
			return_sequences = False
   
		
		#If this is the first layer so stuffs
		if activation != "LeakyReLU":
			if layer_index == 0:
				#n_timesteps, n_features = self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]
				layer = ConvLSTM2D(filters=filters, kernel_size=(1, kernel_size),
									activation=activation,
									return_sequences = return_sequences,
									input_shape=(self.dataset.x_train.shape[1],
												 self.dataset.x_train.shape[2],
												 self.dataset.x_train.shape[3],
												 self.dataset.x_train.shape[4]),
									padding="same")
				these_layers.append(layer)
			else:
				layer = ConvLSTM2D(filters=filters, kernel_size=(1, kernel_size),
										return_sequences = return_sequences,
										activation=activation, padding="same")
				these_layers.append(layer)
		else:
			if layer_index == 0:
				#n_timesteps, n_features = self.dataset.x_train.shape[1], self.dataset.x_train.shape[2]
				layer = ConvLSTM2D(filters=filters, kernel_size=(1, kernel_size),
									return_sequences = return_sequences,
									input_shape=(self.dataset.x_train.shape[1],
												 self.dataset.x_train.shape[2],
												 self.dataset.x_train.shape[3],
												 self.dataset.x_train.shape[4]),
									padding="same")
				these_layers.append(layer)
			else:
				layer = ConvLSTM2D(filters=filters, return_sequences = return_sequences,
									kernel_size=(1, kernel_size), padding="same")
				these_layers.append(layer)
			these_layers.append(LeakyReLU())
	
		print("Add_ConvLSTM2D_Layer not complete and return actual hyperparameters")
		return (bias_init, these_layers)
	
	#Adds the Dropout layer and hyperparameters by condition
	def Add_Dropout_Layer(self, hp, layer_index, all_layer_params):
		name_prefix = "Dropout_" + str(layer_index) + "_"
		
		#layer_parameters = all_layer_params["Dropout"]
				
		if self.good_layer_index == -1:
			#We do this here just because we can
			layer_parameters = all_layer_params["Dropout"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["Dropout_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]
		
		dropout = hp.Choice(name_prefix + "rate", layer_parameters["rate"])
		return Dropout(rate=dropout)
	
	def Add_MaxPooling1D_Layer(self, hp, layer_index, chosen_model, all_layer_params):
		#pool_size is currently a percent, multiply it by window_size to get an int
		name_prefix = "MaxPooling1D_" + str(layer_index) + "_"

		#layer_parameters = all_layer_params["MaxPooling1D"]
				
		if self.good_layer_index == -1:
			#We do this here just because we can
			if "MaxPooling1D" in all_layer_params:
				layer_parameters = all_layer_params["MaxPooling1D"]
			elif "MaxPooling2D" in all_layer_params:
				layer_parameters = all_layer_params["MaxPooling2D"]
			elif "MaxPooling3D" in all_layer_params:
				layer_parameters = all_layer_params["MaxPooling3D"]
		else:
			#Data tuning
			layer_parameters = all_layer_params["MaxPooling1D_"+ str(layer_index)]
			#layer_parameters = all_layer_params[layer_index]

		pool_size = int(self.dataset.window_size*hp.Choice(name_prefix + "pool_size", layer_parameters["pool_size"]))
		
		if "ConvLSTM2D" in chosen_model:
			cl2d_indices = [i for i,d in enumerate(chosen_model) if d == 'ConvLSTM2D']
			
			#if there is a convlstm2d after the current one, use maxpool3d, else maxpool2d
			if len([cli for cli in cl2d_indices if cli > layer_index]) > 0:
				return MaxPooling3D(pool_size = (1, pool_size, pool_size), padding="same")
			else:
				return MaxPooling2D(pool_size = (1, pool_size), padding="same")
		
		return MaxPooling1D(pool_size = pool_size, padding="same")
	
	def Add_Flatten_Layer(self):
		return Flatten()
	
	#==============================================================================
	#START RUNNING THE MODELS
	
	"""
		layer_parameters = Generate_Layer_Parameters()
		layer_depth = Generate_Layer_Depth()
		model_structures = Load_Model_Structures()
	
		#TODO: Select the data parameters to use
		print("TODO: Select the data parameters")
		
		model = The_Model(layer_parameters, model_structures[0])
		model.fit(dataset.x_train, dataset.y_train,
				  validation_data=(dataset.x_test, dataset.y_test),
				  epochs = 3, batch_size = 300)
	"""
	
	#Tunes the model with the given data parameters
	def Tune_Models(self, epochs = 500, batch_size = 256, MAX_TRIALS = 1000):
		EXECUTIONS_PER_TRIAL = 2
		
		#CODE FOR DOING THE BAYESIAN OPTIMIZATION
		tuner = RandomSearch(
				self.The_Model,
				objective = 'val_accuracy',
				max_trials = MAX_TRIALS,
				executions_per_trial = EXECUTIONS_PER_TRIAL,
				directory = 'data\\' + self.parent_folder + '\\',
				project_name = self.folder_name + str(self.folder_suffix) + '\\',
				seed = 42
			)
		
		#CODE FOR DOING THE RANDOM SEARCH
		#tuner = RandomSearch(
		#		self.The_Model,
		#		objective = 'val_accuracy',
		#		max_trials = MAX_TRIALS,
		#		executions_per_trial = EXECUTIONS_PER_TRIAL,
		#		directory = 'data\\' + self.parent_folder + '\\',
		#		project_name = self.folder_name + str(self.folder_suffix) + '\\',
		#		seed = 42
		#	)
		
		#tuner.reload()
		#callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3)]
		#callbacks = [EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, patience = 8, restore_best_weights=True)]
		callbacks = [EarlyStopping(monitor='val_accuracy', mode='max', patience = 8, restore_best_weights=True)]
		
		#strategy = tf.distribute.MirroredStrategy()
		tuner.search(x = self.dataset.x_train,
					 y = self.dataset.y_train,
					 epochs = epochs,
					 batch_size = batch_size,
					 callbacks = callbacks,
					 validation_data = (self.dataset.x_test, self.dataset.y_test))
					 #distribution_strategy = strategy)
		
		#models = tuner.get_best_models(num_models=2)
		
		#Model 0's setup
		#print(models[0].get_config())
		
		return []

#==============================================================================
#==============================================================================

#Generating the model structures can be done separately so its here
#Generate_Model_Strutures(Generate_Layer_Parameters(), Generate_Layer_Depth())

#keras-tuner can not take in an array, it has to be string, int, float
#so we have it select the index of the structure, instead of the actual model
#model_structures = [["GRU"], ["GRU", "GRU"]]# Load_Model_Structures()
#model_structures = [["BidirectionalGRU"], ["BidirectionalGRU", "BidirectionalGRU"]]
#model_structures = [["BidirectionalGRU", "BidirectionalGRU"]]
#model_structures = [["BidirectionalLSTM"], ["BidirectionalLSTM", "BidirectionalLSTM"]]
#model_structures = [["Conv1D", "Flatten"], ["Conv1D", "Conv1D", "Flatten"]]
#model_structures = [["GRU"]]
#model_structures_index = [0]

#data_parameters = Generate_Data_Parameters()
#data_params = {'dataset' : 'firebusters',
#			   'train_p' : 0.8,
#			   'w_size' : 200,
#			   'o_percent' : 0 #0.25
#			   }
#dataset = Load_Data(**data_params)

#mt = Model_Tuning()
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#for device in gpu_devices:
#	tf.config.experimental.set_memory_growth(device, True)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

#Does the hyperparameter tuning of the models on the given dataset
#mt.Tune_Models(dataset)


