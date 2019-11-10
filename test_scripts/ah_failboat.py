#Playng around with graphs
#%%
import numpy as np 
import matplotlib.pyplot as plt 
import csv

# First we need a count of the sample data
def samplecount(proj_folder, data_file):
    with open(proj_folder+data_file, 'r') as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
    n=row_count-1
    return n

#Lets build an array to work with. 
def construct(proj_folder,data_file,col,n):
    da = np.arange(0,n, dtype=float)
    i=0
    with open(proj_folder+data_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            np.put(da,[i],row[col])
            i=i+1
    return da

def samplerate(n,t):
    sr = int(n/(t.max()-t.min()))
    return sr

def main():
    #Select file
    proj_folder = 'C:/Users/andyh/OneDrive/Documents/GitHub/Capstone_Project/data/Captured_Data/Punch/'
    data_file = 'Andy_Punch_rnd1_20throws.csv'

    label = '1st Punch Plot'

    #count samples
    n=samplecount(proj_folder, data_file)

    #read data and create array
    t = construct(proj_folder, data_file,2,n)
    Ax_g = construct(proj_folder,data_file,4,n)

    #convert ms to seconds
    # t = t/1000

    #Calculate sample rate
    sr = float(samplerate(n,t))

    #calculate time step
    dt = float(1.0/sr)

    fig = plt.figure(figsize=(11,6))
    plt.plot(t, Ax_g,'blue')
    plt.grid()
    plt.title('Recorded Accelrometer Data - '+label, fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)',fontsize=14, fontweight='bold')
    plt.ylabel('Acceleration (g)', fontsize=14, fontweight='bold')

    plt.text(tmax(), Ax_g.min()+40, 'Number of Samples: '+str(n),horizontalalignment='right', fontsize=14)
    plt.text(tmax(), Ax_g.min()+20, 'Sample Rate '+str(int(sr))+' samples per second',horizontalalignment='right', fontsize=14)
    plt.text(tmax(), Ax_g.min(), 'Time Step: %.6f seconds' % dt,horizontalalignment='right', fontsize=14)
    
    fig.savefig(proj_folder+'Ax_fig1.png')
    plt.show()

if __name__ == "__main__":
    main()

#%%
def plot_accel(activity, subject, session, data):
	data = data.iloc[500:1000]   									#Select a certain time period
	data = data[data['subject_id']==subject] 						#Filter by person
	data = data[data['session_id']==session] 						#Filter by session
	data = data[data['exercise_id']==activity] 						#filter by exercise

	#Graph all 3 axis on the same plot
	plt.plot( 'TimeStamp_s', 'sID1_AccX_g', data=data, marker='', color='skyblue', linewidth=2, label='X-axis')
	plt.plot( 'TimeStamp_s', 'sID1_AccY_g', data=data, marker='', color='olive', linewidth=2, label="Y-Axis")
	plt.plot( 'TimeStamp_s', 'sID1_AccZ_g', data=data, marker='', color='brown', linewidth=2, label="Z-Axis")
	plt.legend()

    def plot_accel(activity, subject, session, data):
	data = data.iloc[500:1000]   									#Select a certain time period
	data = data[data['subject_id']==subject] 						#Filter by person
	data = data[data['session_id']==session] 						#Filter by session
	data = data[data['exercise_id']==activity] 						#filter by exercise

	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
		figsize=(15, 10),
		sharex=True)
	#Graph all 3 axis on the same plot
	#IMU1
	plot_axis(ax0, data['Time_s'], data['sID1_AccX_g'], 'X-Axis')
	plot_axis(ax0, data['Time_s'], data['sID1_AccY_g'], 'Y-Axis')
	plot_axis(ax0, data['Time_s'], data['sID1_AccZ_g'], 'Z-Axis')

	#IMU2
	plot_axis(ax1, data['Time_s'], data['sID2_AccX_g'], 'X-Axis')
	plot_axis(ax1, data['Time_s'], data['sID2_AccY_g'], 'Y-Axis')
	plot_axis(ax1, data['Time_s'], data['sID2_AccZ_g'], 'Z-Axis')

	#IMU3
	plot_axis(ax2, data['Time_s'], data['sID3_AccX_g'], 'X-Axis')
	plot_axis(ax2, data['Time_s'], data['sID3_AccY_g'], 'Y-Axis')
	plot_axis(ax2, data['Time_s'], data['sID3_AccZ_g'], 'Z-Axis')

	plt.subplots_adjust(hspace=0.2)
	fig.suptitle(activity)
	plt.subplots_adjust(top=0.90)
	plt.xlabel('Time (s)')
	plt.ylabel('Acceleration (g)')
	plt.legend()
	plt.show()

	plt.plot( 'TimeStamp_s', 'sID1_AccX_g', data=data, marker='', color='skyblue', linewidth=2, label='X-axis')
def plot_axis(ax, x, y, label):
    ax.plot(x, y, label = label)
    # ax.set_title(title)
    ax.xaxis.set_visible(True)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

#%%

from numpy import array
from numpy import hstack
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

#%%
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

#%%

# multivariate data preparation
from numpy import array
from numpy import hstack
# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)
    
# trying to understand array reshaping and why it hates me. 

# %%
from numpy import array
 
# load...
data = list()
n = 5000
for i in range(n):
    data.append([i+1, (i+1)*10])
data = array(data)
print(data[:5, :])
print(data.shape)

#%%
# drop time
data = data[:, 1]
print(data.shape)

#%%
samples = list()
length = 200
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
	# grab from i to i + 200
	sample = data[i:i+length]
	samples.append(sample)
print(len(samples))

#%%
# convert list of arrays into 2d array
data = array(samples)
print(data.shape)

#%%
# reshape into [samples, timesteps, features]
# expect [25, 200, 1]
data = data.reshape((len(samples), length, 1))
print(data.shape)

#%%

# load dataset
from numpy import dstack
from pandas import read_csv
import os

os.chdir("C:/githubrepo/CapstoneA/data/HAR/HARdataset/")

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
        print(loaded.shape)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded
 
# load the total acc data
filenames = ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
total_acc = load_group(filenames, prefix='HARDataset/train/Inertial Signals/')
print(total_acc.shape)


# -------------------------Time Window Assignment-------------------------------------
#%%

import os
import pandas as pd
import numpy as np 
import numpy as np
import itertools

from scipy import stats
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import array

os.chdir("C:/githubrepo/CapstoneA/data/")

def load_dataset_windows(df, t_window = 200, t_overlap = 0.25):

	#split by session and exercise and subject
		#iterate through and make windows
		#profit
	#get all exercise ID's
	df_exid = df['exercise_id'].unique()
	#get all subject ID's
	df_subid = df['subject_id'].unique()
	#get all session ID's
	df_sesid = df['session_id'].unique()

	all_combo = list(itertools.product(df_exid, df_subid, df_sesid))
	
	for combo in all_combo:
		df_all = []
		if ((df['exercise_id'] == combo[0]) & (df['subject_id'] == combo[1]) & (df['session_id'] == combo[2])).any():
			#This combination exists, get all rows that match this and add to the dataframe.
			df_all.append( df.loc[(df['exercise_id'] == combo[0]) & (df['subject_id'] == combo[1]) & (df['session_id'] == combo[2])] )
		step = 20
		segments = []
		labels = []
		for i in range(0, len(df_all) - t_window, step):
			xs = df_all['sID1_AccX_g'].values[i: i + t_window]
			ys = df_all['sID1_AccY_g'].values[i: i + t_window]
			zs = df_all['sID1_AccZ_g'].values[i: i + t_window]
			label = stats.mode(df_all['exercise_id'][i: i + t_window])[0][0]
			segments.append([xs, ys, zs])
			labels.append(label)
		
		print(np.array(segments).shape)
	# for a_df in df_all
	# 		nwindows = int(a_df.shape[0]/t_window)
	# 		print(a_df. shape) 
	# 		print(nwindows)
		# Next lowest divisible time stamp with the time window (round down probably)
		# Drop data above that time stamp.
		# Add it to the new array.  
		# Calculate what the overlap is from the t_window
		# iterate to the overlap % (should probably pass that into the load_dataset_windows as a class var)      
		# First window ends at t_window
		#ending after that is
print("Exercise_id, subject_id, session_id")
df = pd.read_csv("Zenshin_Data/ComboPlatter.csv")
val = load_dataset_windows(df)
print(val)


# %%
#---------------------------------------FUCK THIS SHIT-----------------------------------------------------
# Just looking at examples of KNN's
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
X = X[['mean area', 'mean compactness']]
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

sns.scatterplot(
    x='mean area',
    y='mean compactness',
    hue='benign',
    data=X_test.join(y_test, how='outer')
)

plt.scatter(
    X_test['mean area'],
    X_test['mean compactness'],
    c=y_pred,
    cmap='coolwarm',
    alpha=0.7
)

confusion_matrix(y_test, y_pred)