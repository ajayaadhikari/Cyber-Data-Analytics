import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
# columns = StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label


def read_from_file():
    print("Reading from file.")
    source_path = os.path.join("..", "data", "CTU-13-Dataset", "1", "capture20110810.binetflow")
    converters = {"Label": lambda x: 1 if x.startswith("flow=From-Botnet") else (0 if x.startswith("flow=From-Normal") else 2)}
    df = pd.read_csv(source_path, skip_blank_lines=True, delimiter=",", converters=converters)
    df.dropna(subset=["DstAddr", "SrcAddr", "Dport", "Sport", "Label"], inplace=True, how="any")
    print("\tDone!!")
    return df


def read_pickle_file(file_name):
    print("Reading pickle file: %s" % file_name)
    path = os.path.join("..", "data", "pickle", file_name)
    pickle_file = open(path, "r")
    result = pickle.load(pickle_file)
    pickle_file.close()
    print("\tDone!!")
    return result

def save_using_pickle(python_object, name_file):
    pickle_file = open(os.path.join("..", "data", "pickle", name_file), "w")
    pickle.dump(python_object, pickle_file)
    pickle_file.close()

def save_groupings(df, first_column, second_column, name):
    # Get the groupings
    output = nested_grouping(df, first_column, second_column)

    # Write the groupings to file
    print("Writing following grouping: %s, %s" % (first_column, second_column))
    save_using_pickle(output, name)
    print("\tDone!!")


def nested_grouping(df, first_column, second_column):
    print("Performing nested grouping with columns: %s, %s" % (first_column, second_column))
    groupby_fc = df.groupby(first_column)
    result = {}
    fc_values = groupby_fc.groups.keys()
    i = 0
    for fc_value in fc_values:
        # Save the number of occurrences of the current first column value
        groupby_fc_current = groupby_fc.get_group(fc_value)
        result[fc_value] = (len(groupby_fc_current.index), {})

        # Loop over the values of the second column and save their number of occurrences
        groupby_fc_sc = groupby_fc_current.groupby(second_column)
        for sc_value in groupby_fc_sc.groups.keys():
            groupby_fc_sc_current = groupby_fc_sc.get_group(sc_value)
            result[fc_value][1][sc_value] = len(groupby_fc_sc_current.index)
        i += 1
        if i % 200 == 0:
            print("%s/%s, percentage: %s" % (i, len(fc_values), float(i)/len(fc_values)))
    print("\tDone!!")
    return result

def build_feature_vector():
    df = read_from_file()
    # remove the background flows
    df = df[df.Label != 2]

    src_dport = read_pickle_file("src_dport.p")
    dst_sport = read_pickle_file("dst_sport.p")
    feature_vector = []

    for index, row in df.iterrows():
        nr_src = src_dport[row["SrcAddr"]][0]
        nr_dst = dst_sport[row["DstAddr"]][0]

        nr_sport = dst_sport[row["DstAddr"]][1][row["Sport"]]
        nr_dport = src_dport[row["SrcAddr"]][1][row["Dport"]]

        feature_vector.append([nr_src, nr_dst, nr_sport, nr_dport, row["Label"]])
    return feature_vector

def classification_flows():
    feature_vector = np.array(read_pickle_file("feature_vector.p"))
    X = feature_vector[:, 0:4]
    y = [int(x) for x in feature_vector[:, 4:5].flatten().tolist()]

    X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.33)

    classifier = SVC()
    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_test)
    print(accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))


classification_flows()
