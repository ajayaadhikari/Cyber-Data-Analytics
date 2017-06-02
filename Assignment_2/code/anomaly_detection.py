import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import operator
import os
import scipy.stats as stats

#################################################################################
############################## Read data from file ##############################
#################################################################################
attack_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Attack_v0.csv')
normal_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Normal_v0.csv')

converters = {'Normal/Attack': lambda x: "Attack" if x == "A ttack" else x}
attack_data = pd.read_csv(attack_data_path, skip_blank_lines=True, skiprows=1, converters=converters)
normal_data = pd.read_csv(normal_data_path, skip_blank_lines=True, skiprows=1)

#################################################################################
############################## Pre-processing ###################################
#################################################################################

# Remove the whitespaces from the column names
strip = lambda container: map(lambda element: element.strip(), container)
attack_data.columns = strip(attack_data.columns.tolist())
normal_data.columns = strip(normal_data.columns.tolist())

# Create training and testing dataframes
first_attack_row_index = -1
for index, row in attack_data.iterrows():
    if row['Normal/Attack'] == "Attack":
        first_attack_row_index = index
        break

df_before_attack = attack_data.iloc[:first_attack_row_index, :]
training_set = pd.concat([normal_data, df_before_attack])
testing_set = attack_data.iloc[first_attack_row_index:, :]

#################################################################################
############################ PCA-based anomaly detection ########################
#################################################################################


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

#correlation_m =  attack_data.corr(method='pearson')
#correlation_matrix(correlation_m)

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