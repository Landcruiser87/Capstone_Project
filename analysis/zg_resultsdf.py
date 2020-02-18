import itertools
import numpy as np
import pandas as pd
import pickle
import random
import os
import pickle
import seaborn as sns
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.finalcountdown_stage4 import Final_Accuracy

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

#Load in the accuracy files
fa = Final_Accuracy()
fa.Help()
acc = fa.Load_The_File()

#Example printing of original data of each type
print(acc["Dense"][0])
print(acc["GRU"][0])
print(acc["LSTM"][0])
print(acc["BidirectionalGRU"][0])
print(acc["BidirectionalLSTM"][0])
print(acc["Conv1D"][0])
print(acc["ConvLSTM2D"][0])

colnames = ["key", "val_accuracy", "train_accuracy", "depth", "optimizer"]
#key, acc, val_acc, layer depth, optimizer
rows = []
for key in acc:
	for model in acc[key]:
		row = []
		row.append(key)						#key
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

#Removes all the perfect accuracies, because im sure some of them are bad
df = df[df["val_accuracy"] < 1]

#print(df[df["key"] == "GRU"].head())

#print(df[(df["key"] == "GRU") & (df["window_size"] < 50)].head())

#df2 = df[(df["key"] == "GRU") & (df["val_accuracy"] >= 0.8)]

#print(df2.head())
#sns.distplot(df2["val_accuracy"])

#sns.distplot(df["val_accuracy"])

ns.pairplot(df)












