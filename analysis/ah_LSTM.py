#%%
# ms-python.python added
import os
try:
	os.chdir('~/CapstoneA') 
	print(os.getcwd())
except:
	pass


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import timeitnote
from sklearn.model_selection import train_test_split 

#os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")

def split_df(X,y,split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test

def process_target(df,target_col=target_col):
    df[target_col] = (df["exercise_id"]
    return df

def process_drops(df, cols):
    return df.drop(cols,axis=1,inplace=True)


def build_df(drops=None):
    df = pd.read_csv("ComboPlatter.csv", header=TRUE)
    process_target(df, target_col=target_col)
    process_drops(df,drops)
    X = df.drop(columns=["excersie_id",target_col])
    y = df[target_col]
    return X,y


build_df()

