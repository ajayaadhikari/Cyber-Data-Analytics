import pandas as pd
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# columns = StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label

scenario_size = [(1,368), (2,235), (3,610), (4,146), (5,17), (6,73), (7,15), (8,385), (9,272), (11,14), (12,43), (13,250)]
scenarios = [t[0] for t in sorted(scenario_size, key=lambda x: x[1])]
print(scenarios)

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

    # Drop rows that contain null values for atleast one of these rows: "DstAddr", "SrcAddr", "Dport", "Sport", "Label"
    df.dropna(subset=["DstAddr", "SrcAddr", "Dport", "Sport", "Label"], inplace=True, how="any")

    # remove the background flows
    df = df[df.Label != 2]

    print("\tDone!!")
    return df


def save_groupings_scenarios(scenarios):
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
                print("%s/%s, percentage: %s" % (i, len(fc_values), float(i) / len(fc_values)))
        print("\tDone!!")
        return result

    def save_groupings_df(df, first_column, second_column, name):
        # Get the groupings
        output = nested_grouping(df, first_column, second_column)

        # Write the groupings to file
        save_using_pickle(output, name)

    for scenario in scenarios:
        df = read_from_file(scenario)
        save_groupings_df(df, "SrcAddr", "Dport", os.path.join("grouping", "src_dport_%s.p" % scenario))
        save_groupings_df(df, "DstAddr", "Sport", os.path.join("grouping", "dst_sport_%s.p" % scenario))


def read_pickle_file(file_name):
    print("Reading pickle file: %s" % file_name)
    path = os.path.join("..", "data", "pickle", file_name)
    pickle_file = open(path, "r")
    result = pickle.load(pickle_file)
    pickle_file.close()
    print("\tDone!!")
    return result


def save_using_pickle(python_object, file_name):
    print("Writing python object to file: %s" % file_name)
    pickle_file = open(os.path.join("..", "data", "pickle", file_name), "w")
    pickle.dump(python_object, pickle_file)
    pickle_file.close()
    print("\tDone!!")


def save_to_file(content, file_name):
    file_ = open(os.path.join("..", "data", "pickle", file_name), "w")
    file_.write(content)
    file_.close()


def read_txt_file(file_name):
    file_ = open(os.path.join("..", "data", "pickle", file_name), "r")
    data = file_.read()
    file_.close()
    return data

def save_feature_vectors(scenarios):

    def build_feature_vector(scenario):
        df = read_from_file(scenario)

        src_dport = read_pickle_file(os.path.join("grouping", "src_dport_%s.p" % scenario))
        dst_sport = read_pickle_file(os.path.join("grouping", "dst_sport_%s.p" % scenario))
        feature_vector = []

        for index, row in df.iterrows():
            nr_src = src_dport[row["SrcAddr"]][0]
            nr_dst = dst_sport[row["DstAddr"]][0]

            nr_sport = dst_sport[row["DstAddr"]][1][row["Sport"]]
            nr_dport = src_dport[row["SrcAddr"]][1][row["Dport"]]

            feature_vector.append([row["SrcAddr"], nr_src, nr_dst, nr_sport, nr_dport, row["Label"]])
        return feature_vector

    for scenario in scenarios:
        feature_vector = build_feature_vector(scenario)
        save_using_pickle(feature_vector, os.path.join("feature_vector","fv_%s.p" % scenario))


# Type can be "packet_level" or "host_level"
def save_classification_error(scenarios, type):

    def get_evaluation_metrics(real_labels, predicted_labels, pos_label):
        print("Calculating evaluation metrics.")
        accuracy = accuracy_score(real_labels, predicted_labels)
        f1 = f1_score(real_labels, predicted_labels, pos_label=pos_label)
        precision = precision_score(real_labels, predicted_labels,pos_label=pos_label)
        recall = recall_score(real_labels, predicted_labels, pos_label=pos_label)
        confusion_matrix_ = confusion_matrix(real_labels, predicted_labels)

        TP, FP, FN, TN = 0, 0, 0, 0
        for i in xrange(len(predicted_labels)):
            if real_labels[i] == 1 and predicted_labels[i] == 1:
                TP += 1
            if real_labels[i] == 0 and predicted_labels[i] == 1:
                FP += 1
            if real_labels[i] == 1 and predicted_labels[i] == 0:
                FN += 1
            if real_labels[i] == 0 and predicted_labels[i] == 0:
                TN += 1

        resulting_metrics = {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "accuracy": accuracy, "f1": f1,
                             "precision":precision, "recall": recall, "confusion_matrix": confusion_matrix_}
        print("\tDone!!")
        return resulting_metrics

    def evaluate_host_level(real_labels, predicted_labels, host_ips):
        index_host_ips = {}
        for index in range(len(host_ips)):
            if host_ips[index] not in index_host_ips.keys():
                index_host_ips[host_ips[index]] = []
            index_host_ips[host_ips[index]].append(index)

        new_real_labels = []
        new_predicted_labels = []
        for host_ip in host_ips:
            new_real_labels.append("1" if "1" in real_labels[index_host_ips[host_ip]] else "0")
            new_predicted_labels.append("1" if "1" in predicted_labels[index_host_ips[host_ip]] else "0")

        return new_real_labels, new_predicted_labels

    def evaluate(scenario):
        # Read the feature vector for the given scenario
        feature_vector = np.array(read_pickle_file(os.path.join("feature_vector", "fv_%s.p" % scenario)))

        # Get the training set and the labels
        X = feature_vector[:, 1:5]
        y = feature_vector[:, 5:6].flatten()
        host_ips = feature_vector[:, 0:1].flatten()

        # Perform three fold cross validation
        real_labels = []
        predicted_labels = []
        test_host_ips = []
        k_fold = StratifiedKFold(n_splits=3)
        for train, test in k_fold.split(X, y):

            # Build a svm classifier
            classifier = SVC()
            classifier.fit(X[train], y[train])

            real_labels.extend(y[test])
            predicted_labels.extend(classifier.predict(X[test]))
            test_host_ips.extend(host_ips[test])

        pos_label = "1"
        if type == "host_level":
            real_labels, predicted_labels = evaluate_host_level(np.array(real_labels), np.array(predicted_labels), test_host_ips)

        return get_evaluation_metrics(real_labels, predicted_labels, pos_label)

    for scenario in scenarios:
        evaluation_metrics = evaluate(scenario)
        evaluation_metrics_str = reduce(lambda x, y: "%s\n%s" % (x,y), ["%s = %s" % (metric, evaluation_metrics[metric]) for metric in evaluation_metrics.keys()])
        save_to_file(evaluation_metrics_str, os.path.join("evaluation_metric", type, "result_%s.txt" % scenario))

#for scenario in scenarios:
#    save_groupings_scenarios([scenario])
#    save_feature_vectors([scenario])
#    save_classification_error([scenario], "host_level")
#    save_classification_error([scenario], "packet_level")

print(read_from_file(os.path.join("evaluation_metric", "host_level", "result_1.txt" )))