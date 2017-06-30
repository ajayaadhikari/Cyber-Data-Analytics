import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
import math

def read_from_file(scenario):
    print("Reading from file. Scenario: %s" % scenario)

    # Get the path of the file
    dir_path = os.path.join("..", "data", "CTU-13-Dataset", str(scenario))
    file_name = filter(lambda x: x.endswith(".binetflow"), os.listdir(dir_path))[0]
    file_path = os.path.join(dir_path, file_name)

    # Read the csv file in a pandas dataframe
    # Convert label: "flow=From-Botnet" to 1, label: "flow=From-Normal" to 0 and the rest to 2
    converters = {"Label": lambda x: 1 if x.startswith("flow=From-Botnet") else (0 if x.startswith("flow=From-Normal") else 2)}
    df = pd.read_csv(file_path, skip_blank_lines=True, delimiter=",", converters=converters)

    # remove the background flows
    df = df[df.Label != 2]

    # Remove the columns which will not be used the clustering
    # sTos, dTos (Nan and 0 values)

    df.drop(["sTos", "dTos", "State", "DstAddr", "Sport", "Dport"], 1, inplace=True)

    print("\tDone!!")
    return df


def read_scenarios(scenarios):
    result = []
    for scenario in scenarios:
        result.append(read_from_file(scenario))
    df = pd.concat(result)

    # Create dummy types
    df = pd.get_dummies(df, columns=["Proto", "Dir"])

    # Drop rows that contain null values for at-least one of the rows
    df.dropna(inplace=True, how="any")
    return df


def elbow_method(df):
    # convert to numpy array
    used_columns = list(df)
    used_columns.remove("StartTime")
    used_columns.remove("SrcAddr")
    used_columns.remove("Label")
    numpy_array = df.as_matrix(columns=used_columns)
    print(numpy_array[0:5])

    elbow = []
    for k in range(2, 20):
        k_means = KMeans(n_clusters=k).fit(numpy_array)
        elbow.append(k_means.inertia_)

    plt.plot(range(2,20), elbow, "-ro")
    plt.grid(True)
    plt.show()


def split_data(df):
    def groupby_host(df):
        df_groupby_by_src = df.groupby("SrcAddr")
        result = []
        for src in df_groupby_by_src.groups.keys():
            df_groupby_by_src.get_group(src)
            result.append(df_groupby_by_src.get_group(src)["ClusterLabel"].values.tolist())
        return result

    df_group_by_label = df.groupby("Label")
    result = {}
    result[0] = groupby_host(df_group_by_label.get_group(0))
    result[1] = groupby_host(df_group_by_label.get_group(1))
    return result

def discretize(df):
    # remove unneeded columns
    # convert to numpy array
    columns_used = list(set(list(df)) - {"StartTime", "SrcAddr", "Label"})
    num_array = df.as_matrix(columns=columns_used)

    kmeans_labels = KMeans(n_clusters=6).fit(num_array)
    result = pd.DataFrame()
    result["Label"] = df["Label"]
    result["SrcAddr"] = df["SrcAddr"]
    result["ClusterLabel"] = kmeans_labels.labels_

    from collections import Counter
    print(Counter(kmeans_labels.labels_))

    return result

#############################################################################################
####################################### BI-Grams ############################################
#############################################################################################
n = 2

def get_sliding_window(sequence):
    i = 0
    result = []
    while i <= len(sequence) - n:
        result.append(tuple(sequence[i:i+n]))
        i += 1
    return result


def get_ngram_model(sequence):
    records = get_sliding_window(sequence)
    unique_records = set(records)
    result = {}
    for record in unique_records:
        result[record] = records.count(record)/float(len(records))
    return result


def get_probability(ngram_model, sequence):
    records = get_sliding_window(sequence)
    t = Counter(sequence)
    p = 0.0
    for record in records:
        if record in ngram_model.keys():
            p += ngram_model[record]
    normalized_p = p/len(records)
    return normalized_p

def classify(ngram_model, sequence, threshold):
    p = get_probability(ngram_model, sequence)
    if p > threshold:
        return 1
    else:
        return 0

#sequence = "abcdef"
#ngram_model = get_ngram_model(sequence)
#print(classify(ngram_model, "abababababababab"))

def evaluate():
    def predict_threshold(threshold, split):
        num_attack_hosts = len(split[1])
        num_normal_hosts = len(split[0])

        predicted_labels = []
        real_labels = []
        for i in range(num_attack_hosts):
            ngram_model = get_ngram_model(split[1][i])
            for j in range(i) + range(i + 1, num_attack_hosts):
                real_labels.append(1)
                predicted_labels.append(classify(ngram_model, split[1][j], threshold))

            for k in range(num_normal_hosts):
                real_labels.append(0)
                predicted_labels.append(classify(ngram_model, split[0][k], threshold))
        return confusion_matrix(real_labels, predicted_labels)

    df = read_scenarios([11, 12])
    new_df = discretize(df)
    split = split_data(new_df)

    print(predict_threshold(0.99, split))

def plot_hist(data, title):  # fixed number of bins
    plt.hist(data, bins="auto")
    plt.title(title)
    plt.ylabel('Probability')
    plt.show()

def plot_probabilities():

    df = read_scenarios([11, 12])
    new_df = discretize(df)
    split = split_data(new_df)

    num_attack_hosts = len(split[1])
    num_normal_hosts = len(split[0])

    attact_p = []
    normal_p = []
    for i in range(num_attack_hosts):
        ngram_model = get_ngram_model(split[1][i])
        for j in range(i) + range(i+1, num_attack_hosts):
            attact_p.append(get_probability(ngram_model, split[1][j]))

        for k in range(num_normal_hosts):
            normal_p.append(get_probability(ngram_model, split[0][k]))

    plot_hist(attact_p, "Attack distribution")
    plot_hist(normal_p, "Normal distribution")


plot_probabilities()