import itertools
import numpy as np
import pandas as pd
import pickle
import random
import pickle
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import os
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder

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

#==============================================================================

categories = ["Dense", "GRU", "LSTM", "BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D"]
acc = {}
for c in categories:
	with open("./results/" + c + "_s5_results.pkl", 'rb') as f:
		acc[c] = pickle.load(f) 

#Example printing of original data of each type
print(acc["Dense"][0])
#print(acc["GRU"][0])
#print(acc["LSTM"][0])
#print(acc["BidirectionalGRU"][0])
#print(acc["BidirectionalLSTM"][0])
#print(acc["Conv1D"][0])
#print(acc["ConvLSTM2D"][0])

colnames = ["key", "index", "val_accuracy", "test_accuracy", "depth", "optimizer"]
#key, acc, val_acc, layer depth, optimizer
rows = []
for key in acc:
	i = 0
	for model in acc[key]:
		row = []
		row.append(key)						#key
		row.append(i)						#index of where it was in orig
		row.append(statistics.mean(model[0][4]))			#validation accuracy
		row.append(statistics.mean(model[0][2]))			#test accuracy
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
		row.append(model[0][6][0])
		row.append(model[0][6][1])
		row.append(model[0][6][2])
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
#df_best = df[ (abs(df["val_accuracy"] - df["train_accuracy"]) <= 0.3) & (df["val_accuracy"] >= 0.6) ]

#The category of data you want to look at
the_type = "ConvLSTM2D"

#Just looking at "the_type" of data
df_d = df[df["key"] == the_type]
df_d = df_d.sort_values(by=['key', 'val_accuracy'], ascending = False)
print(df_d[["val_accuracy", "test_accuracy", "batch_size", "avg_num_nodes"]].head(20))

#Describing the dataset
print(df_d[["val_accuracy", "test_accuracy", "batch_size", "avg_num_nodes"]].describe())

#Boxplots of validation accuracy vs who it was validating
rows = []
for key in acc:
	for model in acc[key]:
		for i in np.arange(len(model[0][4])):
			row = []
			row.append(key)
			row.append(model[0][4][i])
			row.append(i)
			rows.append(row)

df_val = pd.DataFrame(rows)
df_val.columns = ["key", "val_acc", "index"]
df_d2 = df_val[df_val["key"] == the_type]
df_d2 = df_d2[["val_acc", "index"]]

sns.boxplot(df_d2["index"], df_d2["val_acc"])




