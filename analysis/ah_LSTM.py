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
import timeit
from sklearn.model_selection import train_test_split 

os.chdir("C:/githubrepo/CapstoneA/data/Zenshin_Data/")

def split_df(X,y,split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test


def process_drops(df, cols):
    return df.drop(cols,axis=1,inplace=True)


def build_df(drops=["TimeStamp_s", "exercise_amt", "session_id", "subject_id"]):
    df = pd.read_csv("ComboPlatter.csv")
    process_drops(df,drops)
    y = df["exercise_id"]
    X = df.drop(columns=["exercise_id"])
    return X,y


X,y = build_df()
X_train, X_test, y_train, y_test = split_df(X,y,0.2)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

