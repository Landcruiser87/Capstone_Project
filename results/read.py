import pickle
results  = pickle.load(open('results.pkl', 'rb'))
#print(results)


#Prints out top 100 accuracies
i = 0
for model in results["Conv1D"]:
    print(model[0][3])
    i += 1
    if i > 50:
        break


    
"""
import itertools
import numpy as np
import pandas as pd
import pickle
import random
import os
import pickle
os.chdir("C:/githubrepo/CapstoneA/") #Zack and Andy's github data folder
from analysis.finalcountdown_stage4 import Final_Accuracy

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#Load in the accuracy files
fa = Final_Accuracy()
fa.Help()"""
import pandas as pd
acc = pickle.load(open('results.pkl', 'rb'))

#Sample output
#print(acc["Dense"][0])

#key, acc, val_acc, layer depth, optimizer
rows = []
for key in acc:
	for model in results[key]:
		row = []
		row.append(key)						#key
		row.append(*model[0][3])			#validation accuracy
		row.append(*model[0][1])			#training accuracy
		row.append(len(model[0][0]))		#depth
		row.append(model[1]["optimizer"])	#optimizer
		rows.append(row)

df = pd.DataFrame(rows)
df.columns = ["key", "val_accuracy", "train_accuracy", "depth", "optimizer"]

print(df[df["key"]=="Dense"].head())	
print(df[(df["key"]=="BidirectionalGRU") & (df["val_accuracy"] != 1)].head())
print(df[(df["key"]=="BidirectionalLSTM") & (df["val_accuracy"] != 1)].head())
print(df[(df["key"]=="LSTM") & (df["val_accuracy"] != 1)].head())
print(df[(df["key"]=="GRU") & (df["val_accuracy"] != 1)].head())
print(df[(df["key"]=="Conv1D") & (df["val_accuracy"] != 1)].head())
print(df[(df["key"]=="Conv2D") & (df["val_accuracy"] != 1)].head())




    