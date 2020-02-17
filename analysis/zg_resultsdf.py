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
fa.Help()
acc = fa.Load_The_File()

#Sample output
print(acc["Dense"][0])

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
		rows.append(row)

df = pd.DataFrame(rows)
df.columns = ["key", "val_accuracy", "train_accuracy", "depth", "optimizer"]

print(df.head())	




