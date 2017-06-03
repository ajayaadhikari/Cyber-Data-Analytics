import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize,scale
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
#################################################################################
############################## Read data from file ##############################
#################################################################################
attack_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Attack_v0.csv')
normal_data_path = os.path.join('..', 'data', 'SWaT_Dataset_Normal_v0.csv')

def read_from_file():
    print("Reading Attack and Normal dataset from file.")
    converters = {'Normal/Attack': lambda x: "Attack" if x == "A ttack" else x}
    
    attack_data = pd.read_csv(attack_data_path, skip_blank_lines=True, skiprows=1, converters=converters)
    normal_data = pd.read_csv(normal_data_path, skip_blank_lines=True, skiprows=1)
    print("\t\tDone.")
    return attack_data, normal_data

#################################################################################
############################## Pre-processing ###################################
#################################################################################
training_path = os.path.join('..', 'data', 'training_without_normalization.csv')
testing_path = os.path.join('..', 'data', 'testing_without_normalization.csv')
num_sampels_training = 50000


def plot_axis(df, column_name, range_time):
    values = df[column_name].tolist()[range_time[0]:range_time[1]]
    xmin, xmax, ymin, ymax = 0, len(values), min(values), max(values)
    plt.plot(range(xmax), values, 'ro--')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.show()

def pre_process(attack_df, normal_df):
    print("Pre-processing and creating training and testing sets.")
    # Remove the whitespaces from the column names
    strip = lambda container: map(lambda element: element.strip(), container)
    attack_df.columns = strip(attack_df.columns.tolist())
    normal_df.columns = strip(normal_df.columns.tolist())

    # Create training and testing data frames
    first_attack_row_index = -1
    for index, row in attack_df.iterrows():
        if row['Normal/Attack'] == "Attack":
            first_attack_row_index = index
            break

    df_before_attack = attack_df.iloc[:first_attack_row_index, :]
    training_set = pd.concat([normal_df, df_before_attack])
    testing_set = attack_df.iloc[first_attack_row_index:, :]
    print("\t\tDone.")
    return training_set, testing_set


def write_to_file(training_set, testing_set):
    print("Writing training and testingset to file to avoid recomputation.")
    training_set.to_csv(path_or_buf=training_path, index=False)
    testing_set.to_csv(path_or_buf=testing_path, index=False)
    print("\t\tDone.")


def get_training_testing_data():
    # If training and testing csv files already exists read from them
    if os.path.isfile(training_path) and os.path.isfile(testing_path):
        print("Training and testing sets already exist. Reading from file.")
        training_data = pd.read_csv(training_path, skip_blank_lines=True)
        testing_data = pd.read_csv(testing_path, skip_blank_lines=True)
        print("\t\tDone.")
    # Else read from the original data-set and pre-process it
    # Write the resulting training and testing data-sets to file to avoid re-computation in the future
    else:
        attack_df, normal_df = read_from_file()
        training_data, testing_data = pre_process(attack_df, normal_df)
        write_to_file(training_data, testing_data)
        # you had written 'training_set', 'testing_set'
    return training_data, testing_data

def nomalize_training_set(training_set):
    training_set = training_set.drop(["Normal/Attack", "Timestamp"], axis=1)
    return normalize(training_set.values)

def normalize(np_dataset):
    return (np_dataset - np_dataset.mean(axis=0))/(np_dataset.max(axis=0)-np_dataset.min(axis=0)+0.00000000000001)



def get_sampled_and_normalized_dataset():
    # Get the training and testing sets
    training_set, testing_set = get_training_testing_data()
    print("Sampling %s records." % num_sampels_training)
    training_set = training_set.sample(n=num_sampels_training)
    print("\t\tDone.")

    print("Normalizing.")
    training_set = nomalize_training_set(training_set)
    testing_set, testing_labels = normalize_testing_set(testing_set)
    print("\t\tDone.")
    return training_set, testing_set, testing_labels

def normalize_testing_set(testing_set):
    # Normalize the testing set
    testing_labels = testing_set["Normal/Attack"].tolist()
    testing_set = testing_set.drop(["Normal/Attack", "Timestamp"], axis=1)
    _, sampled_testing_set, _, sampled_testing_labels = train_test_split(testing_set.values, testing_labels, stratify=testing_labels, test_size=0.1)
    return normalize(sampled_testing_set), sampled_testing_labels


def get_sampled_and_normalized_dataset_arma(seconds):
    # Get the training and testing sets
    training_set, testing_set = get_training_testing_data()

    print("Removing actuators' signals")
    columns = list(training_set)
    # for all columns except the label
    # remove actuators: values '1', '2'
    for column in columns[:-1]:
        if len(training_set[column].unique()) < 3:
            del training_set[column]
            del testing_set[column]

    print("Before sampling (training set): %s records." % (training_set.shape,))
    print("Before sampling (testing set): %s records." % (testing_set.shape,))
    print(list(training_set))
    training_set_sampled = training_set.groupby(np.arange(len(training_set)) // seconds).mean()
    testing_set_sampled = testing_set.groupby(np.arange(len(testing_set)) // seconds).mean()

    print("After sampling (training set): %s records." % (training_set_sampled.shape,))
    print("After sampling (testing set): %s records." % (testing_set_sampled.shape,))
    print list(testing_set)
    return training_set_sampled, testing_set_sampled


#################################################################################
############################ Familiarization task ###############################
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
    labels = ['Sex', 'Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings', ]
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()

#training_data, testing_data = get_training_testing_data()
#plot_axis(training_data, "LIT101", (0, 350))

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

#################################################################################
############################ PCA-based anomaly detection ########################
#################################################################################
threshold_high_variability = 0.99
def get_pca(training_data):
    # Get the principal components
    print("Applying PCA!!")
    ipca = IncrementalPCA()
    ipca.fit(training_data)
    print("\t\tDone.")
    return ipca

def get_threshold_index(variance_per_eigenvector, threshold):
    # Get the index of the eigen vector which captures the accumulative variability given by the threshold
    accumulative_variance = map(lambda index : sum(variance_per_eigenvector[:index + 1]), range(len(variance_per_eigenvector)))
    index_threshold = -1
    for index in range(len(accumulative_variance)):
        if accumulative_variance[index] > threshold:
            index_threshold = index
            break
    return index_threshold

def get_high_low_projection_matrices(pca, index_threshold):
    # Get the subspaces with high and low variance
    principle_components = pca.components_[:index_threshold + 1, :]
    high_variance_projection_matrix = np.dot(np.transpose(principle_components), principle_components)
    low_variance_projection_matrix = (np.identity(len(high_variance_projection_matrix)) - high_variance_projection_matrix)
    return high_variance_projection_matrix, low_variance_projection_matrix

def evaluate_pca_anomoly_dectection(training_data, testing_data, testing_labels):
    pca = get_pca(training_data)
    index_threshold = get_threshold_index(pca.explained_variance_ratio_, threshold_high_variability)
    high_variance_projection_matrix, low_variance_projection_matrix = get_high_low_projection_matrices(pca, index_threshold)

    # Get the residuals on the training set
    residuals_training = []
    squared_prediction_error = lambda v: np.square(np.linalg.norm(np.dot(low_variance_projection_matrix, np.transpose(v))))
    for row in training_set:
        SPE = squared_prediction_error(row)
        residuals_training.append(SPE)

    residuals_testing_normal = []
    residuals_testing_attack = []
    for index in range(len(testing_data)):
        SPE = squared_prediction_error(testing_data[index])
        if testing_labels[index] == "Normal":
            residuals_testing_normal.append(SPE)
        else:
            residuals_testing_attack.append(SPE)
    plt.hist(residuals_testing_normal, bins='auto')
    plt.title("Normal test set.")
    plt.show()

    plt.hist(residuals_testing_attack, bins='auto')
    plt.title("Attack test set.")
    plt.show()
    print(min(residuals_testing_normal),max(residuals_testing_normal), len(residuals_testing_normal))
    print(min(residuals_testing_attack),max(residuals_testing_attack),len(residuals_testing_attack))
    print(min(residuals_training),max(residuals_training),len(residuals_training))



#######
# PCA #
#######
#training_set, testing_set, testing_labels = get_sampled_and_normalized_dataset()
3#evaluate_pca_anomoly_dectection(training_set, testing_set, testing_labels)


########
# ARMA #
########

training_set, testing_set = get_sampled_and_normalized_dataset_arma(10)
for i in training_set.iloc[:,-1].round().unique():
    print i


