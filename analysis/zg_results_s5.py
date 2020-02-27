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

#categories = ["Dense", "GRU", "LSTM", "BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D"]
categories = ["Dense", "LSTM", "BidirectionalGRU", "BidirectionalLSTM", "Conv1D", "ConvLSTM2D"]
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



df_d = df[df["key"] == "Dense"]
df_d = df_d.sort_values(by=['key', 'val_accuracy'], ascending = False)
print(df_d[["val_accuracy", "test_accuracy", "batch_size", "avg_num_nodes"]].head(20))

print(df_d[["val_accuracy", "test_accuracy", "batch_size", "avg_num_nodes"]].describe())



#VAL  [0.65402085, 0.7366484, 0.7569586, 0.7145491, 0.76112837, 0.62909293, 0.7233946, 0.6956209, 0.5817551, 0.67078376, 0.69893175, 0.7135816, 0.8172009, 0.65173197, 0.69747627, 0.85378927, 0.7969646, 0.725286, 0.72483516, 0.7041986, 0.62666714, 0.68792313, 0.727321, 0.7521572, 0.7338967, 0.8040567, 0.6212909, 0.6098632]
#TEST [0.5968610698365527, 0.8380146644106035, 0.7463837994214079, 0.47888086642599276, 0.4924565535680183, 0.586472602739726, 0.7314519345831672, 0.8412087098207673, 0.7130554598691009, 0.6383859286083807, 0.7489443378119002, 0.5925369933519193, 0.3602969166349448, 0.5846422338568935, 0.4988640666414237, 0.639744824265768, 0.7471320717717423, 0.6908971121115626, 0.7177318295739349, 0.7543960464314446, 0.816782578953369, 0.6421096259454979, 0.7552730956986277, 0.6860406091370559, 0.5524208921082729, 0.47478085931886765, 0.4167401991680323, 0.2818829290006677]


rows = []
for key in acc:
	i = 0
	for model in acc[key]:
		row = []
		row.append(key)						#key
		row.extend(model[0][4])
		rows.append(row)

df_val = pd.DataFrame(rows)
df_d2 = df_val[df_val[0] == "Dense"]

sns.boxplot(df_d2)




