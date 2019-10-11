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
