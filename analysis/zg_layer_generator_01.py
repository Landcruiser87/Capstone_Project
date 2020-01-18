"""
		This script generates the possible neural network layers and creates
		the .pkl files of the lists of network layers by category.
"""

from itertools import product
import numpy as np
import pickle
import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#START SETTING UP THE PARAMETERS/LAYERS FOR THE MODELS

class Layer_Generator:
	
	def __init__(self, m_tuning = "all", m_layers = []):
		self.model_tuning = m_tuning
		self.model_layer = m_layers
		return
	
	#Generates the possible layer parameters
	def Generate_Layer_Parameters(self):
		
		#This gets called in the case of layer tuning, not hyperparmeter tuning
		if self.model_tuning != "all":
			if self.model_layer != []:
				return self.Generate_Specific_Layer_Parameters()
			return self.Generate_Simple_Layer_Parameters()
		
		layer_parameters = {}
		
		units = [10, 25, 50, 100, 250, 500]
		activation = ["relu", "tanh", "LeakyReLU"]
		bias_init = ["Zeros", "RandomNormal", "glorot_normal"]
		dropout = [0.0, 0.25, 0.5]
		
		layer_parameters["GRU"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init, #"return_sequences" : [True, False],
									"dropout" : dropout}
		layer_parameters["LSTM"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init, #"return_sequences" : [True, False],
									"dropout" : dropout}
		layer_parameters["Dense"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init}
		layer_parameters["BidirectionalLSTM"] = {"layer" : ["LSTM"]}
		layer_parameters["BidirectionalGRU"] = {"layer" : ["GRU"]}
	
		layer_parameters["Conv1D"] = {"filters" : [0.25, 0.5, 0.75],
										"activation" : activation}#, "kernel_size" : filters*kernel_size = window_size?
		#print("TODO: Conv1D/ConvLSTM2D filters and kernel_size")
		layer_parameters["ConvLSTM2D"] = {"filters" : [0.25, 0.5, 0.75],
										"activation" : activation,
										"n_steps" : [4],
										"dropout" : dropout}#, "kernel_size" : filters*kernel_size = window_size?
		#print("TODO: filters is currently a percentage of window size, has to be an int at the end")
		#FAUX LAYERS
		layer_parameters["Dropout"] = {"rate" : [0.2, 0.35, 0.5]}
		layer_parameters["MaxPooling1D"] = {"pool_size" : [0.1, 0.2, 0.25]}
		#print("TODO: we want strides later... maybe")
		#print("TODO: MaxPooling1D - pool_size has to be an integer in the end (based on window size)")
		layer_parameters["Flatten"] = {}
		
		#print("TODO: Had to remove print in the Invalid_LayerName area, earch the file for 'TODO'")
		return layer_parameters
	
	#Returns the hyperparameters for the layers that need specific parameters
	def Generate_Specific_Layer_Parameters(self):
		layer_parameters = {}
		
		#The layers used in the model
		self.model_layer
		
		#Find the hyperparameters used with these layers
		#There could be multiple layer setups with the same layers so we need
		#to somehow mark off/eliminate the ones that we have already used
		
		units = [10]
		activation = ["relu"]
		bias_init = ["Zeros"]
		dropout = [0.25]
		
		layer_parameters["GRU"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init, #"return_sequences" : [True, False],
									"dropout" : dropout}
		layer_parameters["LSTM"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init, #"return_sequences" : [True, False],
									"dropout" : dropout}
		layer_parameters["Dense"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init}
		layer_parameters["BidirectionalLSTM"] = {"layer" : ["LSTM"]}
		layer_parameters["BidirectionalGRU"] = {"layer" : ["GRU"]}
	
		layer_parameters["Conv1D"] = {"filters" : [0.75],
										"activation" : activation}
		layer_parameters["ConvLSTM2D"] = {"filters" : [0.75],
										"activation" : activation,
										"n_steps" : [4],
										"dropout" : dropout}
		#FAUX LAYERS
		layer_parameters["Dropout"] = {"rate" : [0.2]}
		layer_parameters["MaxPooling1D"] = {"pool_size" : [0.25]}
		#print("TODO: we want strides later... maybe")
		#print("TODO: MaxPooling1D - pool_size has to be an integer in the end (based on window size)")
		layer_parameters["Flatten"] = {}
		
		#print("TODO: Had to remove print in the Invalid_LayerName area, earch the file for 'TODO'")
		return layer_parameters
		print("Generate_Specific_Layer_Parameters is not finished")
		return
	
	#These are for figuring out what layer ordering is the best, not
	#for hyperparameter tuning
	def Generate_Simple_Layer_Parameters(self):
		layer_parameters = {}
		
		units = [10]
		activation = ["relu"]
		bias_init = ["Zeros"]
		dropout = [0.25]
		
		layer_parameters["GRU"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init, #"return_sequences" : [True, False],
									"dropout" : dropout}
		layer_parameters["LSTM"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init, #"return_sequences" : [True, False],
									"dropout" : dropout}
		layer_parameters["Dense"] = {"units" : units,
									"activation" : activation,
									"bias_initializer" : bias_init}
		layer_parameters["BidirectionalLSTM"] = {"layer" : ["LSTM"]}
		layer_parameters["BidirectionalGRU"] = {"layer" : ["GRU"]}
	
		layer_parameters["Conv1D"] = {"filters" : [0.75],
										"activation" : activation}
		layer_parameters["ConvLSTM2D"] = {"filters" : [0.75],
										"activation" : activation,
										"n_steps" : [4],
										"dropout" : dropout}
		#FAUX LAYERS
		layer_parameters["Dropout"] = {"rate" : [0.2]}
		layer_parameters["MaxPooling1D"] = {"pool_size" : [0.25]}
		#print("TODO: we want strides later... maybe")
		#print("TODO: MaxPooling1D - pool_size has to be an integer in the end (based on window size)")
		layer_parameters["Flatten"] = {}
		
		#print("TODO: Had to remove print in the Invalid_LayerName area, earch the file for 'TODO'")
		return layer_parameters
	
	def Generate_Layer_Depth(self):
		 return [1, 2, 3, 4, 5, 6, 7]
	
	#If you have created the model structure already, this loads it in from a file
	def Load_Model_Structures(self, name = "All_Model_Structures"):
		model_structures = []
		
		# open file and read the content in a list
		with open("data/" + name + ".pkl", "rb") as fp:   # Unpickling
			model_structures = pickle.load(fp)
		
		return model_structures
	
	#Deletes all of the invalid layer structures
	def Delete_Invalid_Model_Structures(self, all_models):
		itr = 0
		#Loop through the index of each model (in reverse order)
		for index in np.arange(len(all_models))[::-1]:
			#A ton of if statements that are checked
			#If any of the statements are true we delete
			#the whole model from all_models
			
			#Deals with GRU--------------------------------------------------------
			if self.Invalid_GRU(all_models[index]):
				del all_models[index]
				continue
			#print("1")
			
			#Deals with LSTM-------------------------------------------------------
			if self.Invalid_LSTM(all_models[index]):
				del all_models[index]
				continue
			#print("2")
			
			#Deals with Dense------------------------------------------------------
			if self.Invalid_Dense(all_models[index]):
				del all_models[index]
				continue
			#print("3")
			
			#Deals with Bidirectional LSTM-----------------------------------------
			if self.Invalid_BidirectionalLSTM(all_models[index]):
				del all_models[index]
				continue
			#print("4")
			
			#Deals with Bidirectional GRU------------------------------------------
			if self.Invalid_BidirectionalGRU(all_models[index]):
				del all_models[index]
				continue
			#print("5")
			
			#Deals with Conv1D-----------------------------------------------------
			if self.Invalid_Conv1D(all_models[index]):
				del all_models[index]
				continue
			#print("6")
					
			#Deals with ConvLSTM2D-------------------------------------------------
			if self.Invalid_ConvLSTM2D(all_models[index]):
				del all_models[index]
				continue
			#print("7")
					
			#Deals with Dropout----------------------------------------------------
			if self.Invalid_Dropout(all_models[index]):
				del all_models[index]
				continue
			#print("8")
					
			#Deals with MaxPooling1D----------------------------------------------------
			if self.Invalid_MaxPooling1D(all_models[index]):
				del all_models[index]
				continue
			#print("9")
					
			#Flatten---------------------------------------------------------------
			if self.Invalid_Flatten(all_models[index]):
				del all_models[index]
				continue
			#print("10")
			
			if itr%5000 == 0:
				print(str(itr) + ", ", end="")
			itr = itr + 1
		
		return all_models
	
	#Generates all of the structures possible for the layers in a model then
	#saves it to a file
	def Generate_Model_Strutures(self, layer_p, depth):
		all_models = []
		
		#Get all the layer names
		layer_names = [key for key in layer_p]
		
		#This will be used for making all the permutations of layers
		list_o_layers = []
		for _ in np.arange(max(depth)):
			list_o_layers.append(layer_names)
		
		#Makes all the layer permutations for each desired depth
		for d in depth:
			if d == 1:
				all_models.extend( list(map(lambda el:[el], list_o_layers[0])) )
			else:
				all_models.extend( list(product(*list_o_layers[0:d])) )
	
		#models_list = all_models
		models_list = self.Delete_Invalid_Model_Structures(all_models)
		models_list = [list(ele) for ele in models_list] 
		
		with open("data/All_Model_Structures.pkl", "wb") as fp:   #Pickling
			pickle.dump(models_list, fp)
	
		return []
	
	def Invalid_GRU(self, model):
		#print("TODO: Invalid_GRU, check if bidirectional before gru is valid")
		gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
		
		if len(gru_indices) > 0:
			#No dense before gru
			dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
			for gi in gru_indices:
				if len([di for di in dense_indices if di < gi]) > 0:
					return True
			
			#No flatten before GRU
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for gi in gru_indices:
				if len([fi for fi in flatten_indices if fi < gi]) > 0:
					return True
				
			#No CNN types before GRU
			conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
			for gi in gru_indices:
				if len([fi for fi in conv1d_indices if fi < gi]) > 0:
					return True
			convlstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
			for gi in gru_indices:
				if len([fi for fi in convlstm_indices if fi < gi]) > 0:
					return True
	
			#Bidir GRU before GRU	   Maybe?  TODO: LOOK AT LATER
			#CNN before gru			 NO
			#ConvLSTM before gru?	   NO
			#maxpooling1d before gru	TRUE
	
		return False
	
	def Invalid_LSTM(self, model):
		#print("TODO: Invalid_LSTM, check if bidirectional before gru is valid")
		lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
		
		if len(lstm_indices) > 0:
			#No dense before lstm
			dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
			for li in lstm_indices:
				if len([di for di in dense_indices if di < li]) > 0:
					return True
			
			#No flatten before LSTM
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for li in lstm_indices:
				if len([fi for fi in flatten_indices if fi < li]) > 0:
					return True
				
			#No CNN types before LSTM
			conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
			for li in lstm_indices:
				if len([fi for fi in conv1d_indices if fi < li]) > 0:
					return True
			convlstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
			for li in lstm_indices:
				if len([fi for fi in convlstm_indices if fi < li]) > 0:
					return True
	
			#Bidir GRU before lstm	   Maybe?  TODO: LOOK AT LATER
			#CNN before lstm			 NO
			#ConvLSTM before lstm?	   NO
			#maxpooling1d before lstm	TRUE
	
		return False
	
	def Invalid_Dense(self, model):
		dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
		
		if len(dense_indices) > 0:
			#If Conv1D or ConvLSTM2D are before it, must have flatten
			conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
			convlstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
			cnn_indices = (conv1d_indices + convlstm_indices).sort()
			
			#Has to be at least one cnn type for this case to matter
			if len(cnn_indices) > 0:
				flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
	
				first_dense = dense_indices[0]
				last_cnn = cnn_indices[-1]
				#If there is not flatten between cnn and dense this is invalid
				if len([fi for fi in flatten_indices if fi < first_dense and fi > last_cnn]) == 0:
					return True
		
		return False
	
	def Invalid_BidirectionalLSTM(self, model):
		blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
		
		if len(blstm_indices) > 0:
			#No dense before blstm
			dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
			for bli in blstm_indices:
				if len([di for di in dense_indices if di < bli]) > 0:
					return True
				
			#No flatten before BLSTM
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for bli in blstm_indices:
				if len([fi for fi in flatten_indices if fi < bli]) > 0:
					return True
				
			#No CNN types before BidirectionalLSTM
			conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
			for bli in blstm_indices:
				if len([fi for fi in conv1d_indices if fi < bli]) > 0:
					return True
			convlstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
			for bli in blstm_indices:
				if len([fi for fi in convlstm_indices if fi < bli]) > 0:
					return True
	
			#CNN before it			 NO
			#ConvLSTM before it?	   NO
			#maxpooling1d before it	TRUE
			#Bidir before it		   TRUE
	
		return False
	
	def Invalid_BidirectionalGRU(self, model):
		bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
		
		if len(bgru_indices) > 0:
			#No dense before bgru
			dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
			for bgi in bgru_indices:
				if len([di for di in dense_indices if di < bgi]) > 0:
					return True
			
			#No flatten before BGRU
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for bgi in bgru_indices:
				if len([fi for fi in flatten_indices if fi < bgi]) > 0:
					return True
				
			#No CNN types before BidirectionalGRU
			conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
			for bgi in bgru_indices:
				if len([fi for fi in conv1d_indices if fi < bgi]) > 0:
					return True
			convlstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
			for bgi in bgru_indices:
				if len([fi for fi in convlstm_indices if fi < bgi]) > 0:
					return True
	
			#CNN before it			 NO
			#ConvLSTM before it?	   NO
			#maxpooling1d before it	TRUE
			#Bidir before it		   TRUE
	
		return False
	
	def Invalid_Conv1D(self, model):
		conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
		
		if len(conv1d_indices) > 0:
			#No RNN style before it
			gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
			for ci in conv1d_indices:
				if len([fi for fi in gru_indices if fi < ci]) > 0:
					return True
	
			lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
			for ci in conv1d_indices:
				if len([fi for fi in lstm_indices if fi < ci]) > 0:
					return True
	
			blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
			for ci in conv1d_indices:
				if len([fi for fi in blstm_indices if fi < ci]) > 0:
					return True
	
			bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
			for ci in conv1d_indices:
				if len([fi for fi in bgru_indices if fi < ci]) > 0:
					return True
	
			#No flatten before it
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for ci in conv1d_indices:
				if len([fi for fi in flatten_indices if fi < ci]) > 0:
					return True
			
			#Must have at least one flatten
			if len(flatten_indices) == 0:
				return True
			
			#No dense before it
			dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
			for ci in conv1d_indices:
				if len([di for di in dense_indices if di < ci]) > 0:
					return True
			
			#Conv1D, MaxPooling1D, Dropout, Flatten, Dense
			
		return False
	
	def Invalid_ConvLSTM2D(self, model):
		clstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
		
		if len(clstm_indices) > 0:
			#No RNN style before it
			gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
			for cli in clstm_indices:
				if len([fi for fi in gru_indices if fi < cli]) > 0:
					return True
	
			lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
			for cli in clstm_indices:
				if len([fi for fi in lstm_indices if fi < cli]) > 0:
					return True
	
			blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
			for cli in clstm_indices:
				if len([fi for fi in blstm_indices if fi < cli]) > 0:
					return True
	
			bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
			for cli in clstm_indices:
				if len([fi for fi in bgru_indices if fi < cli]) > 0:
					return True
	
			#No flatten before it
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for cli in clstm_indices:
				if len([fi for fi in flatten_indices if fi < cli]) > 0:
					return True

			#Must have at least one flatten
			if len(flatten_indices) == 0:
				return True
	
			#No dense before it
			dense_indices = [i for i,d in enumerate(model) if d == 'Dense']
			for cli in clstm_indices:
				if len([di for di in dense_indices if di < cli]) > 0:
					return True
			
		return False
	
	#If the way Dropout is set in the model is invalid, it markes it for removal
	def Invalid_Dropout(self, model):
		#Gets the indicies of all the rows that contain dropout
		dropout_indices = [i for i,d in enumerate(model) if d == 'Dropout']
	
		if len(dropout_indices) > 0:
			#First layer can't be dropout
			if dropout_indices[0] == 0:
				return True
	
			#Eliminates dropout two in a row
			for i in np.arange(len(dropout_indices)):
				if i > 0:
					#if any of the indices are next to each other, delete
					if dropout_indices[i] == (dropout_indices[i-1]+1):
						return True
	
			#Dropout - faux layer - dropout is INVALID
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			pool_indices = [i for i,d in enumerate(model) if d == 'MaxPooling1D']
			faux_indices = list(flatten_indices + pool_indices)#.sort()
			faux_indices.sort()
			if len(faux_indices) > 0:
				for di in dropout_indices:
					if (di-1) in faux_indices:
						if (di-2) in dropout_indices:
							return True
						if (di-2) in faux_indices:
							if (di-3) in dropout_indices:
								return True
					
		#This layer is fine
		return False
	
	def Invalid_MaxPooling1D(self, model):
		#print("TODO: Invalid_MaxPooling1D, check if it can be the first layer")
		pool_indices = [i for i,d in enumerate(model) if d == 'MaxPooling1D']
		
		if len(pool_indices) > 0:
			#First layer can't be MaxPoolin1D
			if pool_indices[0] == 0:
				return True
			
			#Two pool in a row is bad
			for i in np.arange(len(pool_indices)):
				if i > 0:
					#if any of the indices are next to each other, delete
					if pool_indices[i] == (pool_indices[i-1]+1):
						return True	   
			
			#Can't be after flatten
			flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
			for pi in pool_indices:
				if len([fi for fi in flatten_indices if fi < pi]) > 0:
					return True
			
		return False
	
	def Invalid_Flatten(self, model):
		
		#Gets the indicies of all the rows that contain dropout
		flatten_indices = [i for i,d in enumerate(model) if d == 'Flatten']
		
		if len(flatten_indices) > 0:
			#First layer can't be flatten------------------------------------------
			if flatten_indices[0] == 0:
				return True
			#We only want 1 flatten per model structure----------------------------
			if len(flatten_indices) > 1:
				return True
	
			gru_indices = [i for i,d in enumerate(model) if d == 'GRU']
			lstm_indices = [i for i,d in enumerate(model) if d == 'LSTM']
			conv1d_indices = [i for i,d in enumerate(model) if d == 'Conv1D']
			blstm_indices = [i for i,d in enumerate(model) if d == 'BidirectionalLSTM']
			bgru_indices = [i for i,d in enumerate(model) if d == 'BidirectionalGRU']
			clstm_indices = [i for i,d in enumerate(model) if d == 'ConvLSTM2D']
			
			#Flatten can not be before CNN or RNN type-----------------------------
			#This gets the indices of each CNN/RNN type layer and then checks to see
			#if one of the indices are greater than the location of the flatten index
			for fi in flatten_indices:
				if len([i for i in gru_indices if i > fi]) > 0:
					return True
				if len([i for i in lstm_indices if i > fi]) > 0:
					return True
				if len([i for i in conv1d_indices if i > fi]) > 0:
					return True
				if len([i for i in blstm_indices if i > fi]) > 0:
					return True
				if len([i for i in bgru_indices if i > fi]) > 0:
					return True
				if len([i for i in clstm_indices if i > fi]) > 0:
					return True
			
			#Flatten can only be immediately after CNN/ConvLSTM--------------------
			#There can be dropout or max pooling before it also, but the CNN type
			#has to then be before that
			drop_indices = [i for i,d in enumerate(model) if d == 'Dropout']
			pool_indices = [i for i,d in enumerate(model) if d == 'MaxPooling1D']
			flag = True #True means this is an bad type
			#Prev layer is CNN type
			if ((flatten_indices[0]-1) in conv1d_indices) or ((flatten_indices[0]-1) in clstm_indices):
				flag = False
			#2 layers back is CNN type
			if ((flatten_indices[0]-2) in conv1d_indices) or ((flatten_indices[0]-2) in clstm_indices):
				#AND 1 layer back is dropout or max pool
				if ((flatten_indices[0]-1) in drop_indices) or ((flatten_indices[0]-1) in pool_indices):
					flag = False
			#3 layers back is CNN type
			if ((flatten_indices[0]-3) in conv1d_indices) or ((flatten_indices[0]-3) in clstm_indices):
				#AND 2 layers back is dropout or max pool
				if ((flatten_indices[0]-2) in drop_indices) or ((flatten_indices[0]-2) in pool_indices):
					#AND 1 layer back is dropout or max pool
					if ((flatten_indices[0]-1) in drop_indices) or ((flatten_indices[0]-1) in pool_indices):
						flag = False
			#If none of these cases occur, then flatten is in an invalid location
			if flag:
				return True
	
			#CNN, Flat			   Dense
			#CNN, Drop, Flat		 Dense
			#CNN, Flat, Drop		 Dense 
			#CNN, Pool, Flat		 Dense
			#CNN, Pool, Flat, Drop   Dense
			#CNN, Pool, Drop, Flat   Dense
			#CNN, Drop, Pool, Flat   Dense
		return False
	
	def Is_GRU_Type(self, model):
		ok_layers = ["GRU", "Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "GRU" in model:
			return True
		return False
	
	def Is_LSTM_Type(self, model):
		ok_layers = ["LSTM", "Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "LSTM" in model:
			return True
		return False
	
	def Is_Dense_Type(self, model):
		ok_layers = ["Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "Dense" in model:
			return True
		return False
	
	def Is_BidirectionalLSTM_Type(self, model):
		ok_layers = ["BidirectionalLSTM", "Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "BidirectionalLSTM" in model:
			return True
		return False
	
	def Is_BidirectionalGRU_Type(self, model):
		ok_layers = ["BidirectionalGRU", "Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "BidirectionalGRU" in model:
			return True
		return False
	
	def Is_Conv1D_Type(self, model):
		ok_layers = ["Conv1D", "Flatten", "MaxPooling1D", "Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "Conv1D" in model:
			return True
		return False
	
	def Is_ConvLSTM2D_Type(self, model):
		ok_layers = ["ConvLSTM2D", "Flatten", "MaxPooling1D", "Dense", "Dropout", "LeakyReLU"]
		for layer in model:
			if layer not in ok_layers:
				return False
		if "ConvLSTM2D" in model:
			return True
		return False
	
	def Save_Models(self, models_list, name):
		with open("data/" + name + ".pkl", "wb") as fp:
			pickle.dump(models_list, fp)
		return
	
	#Loads the data from the .pkl file and splits it into files of different
	#neural network categories
	def Split_Save_Models_Into_Categories(self):
		#Import massive list of all the models
		all_models = self.Load_Model_Structures()
	
		#Lists to hold each category of model setup
		gru_models = []
		lstm_models = []
		dense_models = []
		blstm_models = []
		bgru_models = []
		conv1d_models = []
		convlstm_models = []
		other_models = []
		
		#Iterate through the list of models
		for model in all_models:
			#Checks the model to see if it is a GRU model
			if self.Is_GRU_Type(model):
				gru_models.append(model)
				continue
			#Checks the model to see if it is a LSTM model
			if self.Is_LSTM_Type(model):
				lstm_models.append(model)
				continue
			#Checks the model to see if it is a Dense model
			if self.Is_Dense_Type(model):
				dense_models.append(model)
				continue
			#Checks the model to see if it is a BidirectionalLSTM model
			if self.Is_BidirectionalLSTM_Type(model):
				blstm_models.append(model)
				continue
			#Checks the model to see if it is a BidirectionalGRU model
			if self.Is_BidirectionalGRU_Type(model):
				bgru_models.append(model)
				continue
			#Checks the model to see if it is a Conv1D model
			if self.Is_Conv1D_Type(model):
				conv1d_models.append(model)
				continue
			#Checks the model to see if it is a ConvLSTM2D model
			if self.Is_ConvLSTM2D_Type(model):
				convlstm_models.append(model)
				continue
	
			#If we have reached this point without it being added to any of the
			#other categories then it is categorized as 'other'
			other_models.append(model)
		
		self.Save_Models(gru_models, "GRU_Model_Structures")
		self.Save_Models(lstm_models, "LSTM_Model_Structures")
		self.Save_Models(dense_models, "Dense_Model_Structures")
		self.Save_Models(blstm_models, "BidirectionalLSTM_Model_Structures")
		self.Save_Models(bgru_models, "BidirectionalGRU_Model_Structures")
		self.Save_Models(conv1d_models, "Conv1D_Model_Structures")
		self.Save_Models(convlstm_models, "ConvLSTM2D_Model_Structures")
		self.Save_Models(other_models, "Other_Model_Structures")
		
		return

#Makes an instance of the class.. not sure it necessary
#gen = Layer_Generator()

#Generates and saves to disk the huge list of all model layer setups
#gen.Generate_Model_Strutures(gen.Generate_Layer_Parameters(), gen.Generate_Layer_Depth())

#Saves and splits up that huge list of all saved layer setups into categories
#gen.Split_Save_Models_Into_Categories()

#Examples of loading in a layer structure
#models = gen.Load_Model_Structures("Conv1D_Model_Structures")
#models = gen.Load_Model_Structures("All_Model_Structures")
#models = gen.Load_Model_Structures("Other_Model_Structures")

#print(models)
#print(len(models))



