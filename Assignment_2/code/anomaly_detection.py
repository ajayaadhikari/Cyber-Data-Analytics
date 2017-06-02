import pandas as pd
import os
import matplotlib.pyplot as plt
#################################################################################
############################## Read data from file ##############################
#################################################################################
attack_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Attack_v0.csv')
normal_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Normal_v0.csv')

def read_from_file():
    converters = {'Normal/Attack': lambda x: "Attack" if x == "A ttack" else x}
    attack_data = pd.read_csv(attack_data_path, skip_blank_lines=True, skiprows=1, converters=converters)
    normal_data = pd.read_csv(normal_data_path, skip_blank_lines=True, skiprows=1)
    return attack_data, normal_data

#################################################################################
############################## Pre-processing ###################################
#################################################################################
training_path = os.path.join('..', 'data', 'training.csv')
testing_path = os.path.join('..', 'data', 'testing.csv')

def plot_axis(df, column_name, range_time):
    values = df[column_name].tolist()[range_time[0]:range_time[1]]
    xmin, xmax, ymin, ymax = 0, len(values), min(values), max(values)
    plt.plot(range(xmax), values, 'ro')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.show()

def pre_process(attack_df, normal_df):
    # Remove the whitespaces from the column names
    strip = lambda container: map(lambda element: element.strip(), container)
    attack_df.columns = strip(attack_df.columns.tolist())
    normal_df.columns = strip(normal_df.columns.tolist())

    # Create training and testing dataframes
    first_attack_row_index = -1
    for index, row in attack_df.iterrows():
        if row['Normal/Attack'] == "Attack":
            first_attack_row_index = index
            break

    df_before_attack = attack_df.iloc[:first_attack_row_index, :]
    training_set = pd.concat([normal_df, df_before_attack])
    testing_set = attack_df.iloc[first_attack_row_index:, :]
    return training_set,testing_set


def write_to_file(training_set, testing_set):
    training_set.to_csv(path_or_buf=training_path, index=False)
    testing_set.to_csv(path_or_buf=testing_path, index=False)


def get_training_testing_data():
    # If training and testing csv files already exists read from them
    if os.path.isfile(training_path) and os.path.isfile(testing_path):
        training_data = pd.read_csv(training_path, skip_blank_lines=True)
        testing_data = pd.read_csv(testing_path, skip_blank_lines=True)
    # Else read from the original data-set and pre-process it
    # Write the resulting training and testing datasets to file to avoid recomputation in the future
    else:
        attack_df, normal_df = read_from_file()
        training_data, testing_data = pre_process(attack_df, normal_df)
        write_to_file(training_data, testing_data)

    return training_data, testing_data


training_data, testing_data = get_training_testing_data()
plot_axis(training_data, "MV302", (0,300))

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