import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import operator
import os
import scipy.stats as stats


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


attack_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Attack_v0.csv')
normal_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Normal_v0.csv')

attack_data = pd.read_csv(attack_data_path, skip_blank_lines=True, skiprows=1)
normal_data = pd.read_csv(normal_data_path, skip_blank_lines=True, skiprows=1)

print attack_data.shape
print normal_data.shape

print attack_data['Normal/Attack'].value_counts().to_dict()
print normal_data['Normal/Attack'].value_counts().to_dict()

correlation_m =  attack_data.corr(method='pearson')
correlation_matrix(correlation_m)

#print len(correlation_matrix)
#print correlation_matrix


#for i in range(len(attack_data)-1):
#    print attack_data.ix[i+1].name
#    print stats.normaltest(attack_data.ix[:,i+1 ])
#pd.plot.hist(by=)
#signals = data.drop(data.columns[0], 1) #remove Timestamp
#signals = signals.drop('Normal/Attack', 1)


#print data.ix[:,2]
#correlation = np.corrcoef(data.columns[2]), data.columns[3])