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