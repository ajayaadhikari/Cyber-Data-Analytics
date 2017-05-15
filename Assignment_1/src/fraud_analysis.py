from __future__ import division
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import operator
import os


data_path = os.path.join('..', 'datasets', 'data_for_student_case.csv')
columns = ["txid", "bookingdate", "issuercountrycode", "txvariantcode", "card_issuer_identifier",
           "amount", "currencycode", "shoppercountrycode", "shopperinteraction", "simple_journal",
           "cardverificationcodesupplied", "cvcresponsecode", "creationdate", "accountcode", "mail_id",
           "ip_id", "card_id"]

selected_features = ["issuercountrycode", "txvariantcode", "amount", "shopperinteraction", "cardverificationcodesupplied",
                     "cvcresponsecode", "simple_journal"]
label = "simple_journal"
features_for_convertion = ["issuercountrycode", "txvariantcode", "shopperinteraction"]

class Fraud:
    def __init__(self):
        self.load_data()
        self.run()

    def load_data(self):
        # Read from csv
        print("Reading from file")
        self.df = pd.read_csv(data_path, skip_blank_lines=True)
        print("\tFinished!!!")

        print("Pre-processing.")
        # Rename a column
        self.df.rename(columns={'bin': 'card_issuer_identifier'}, inplace=True)
        # Delete rows with null values for card_issuer_identifier
        self.df = self.df[pd.notnull(self.df.card_issuer_identifier)]
        self.df = self.df[self.df.simple_journal != "Refused"]

        # Change data types of some columns
        self.df["bookingdate"].apply(self.string_to_timestamp)
        self.df["card_issuer_identifier"].apply(float)
        self.df["amount"].apply(lambda x: float(x)/100)
        print("\tFinished!!")
        #self.df["simple_journal"].apply(lambda y: 0 if y == "Settled" else 1)

    # The minority class gets oversampled to balance with the majority class
    # Output format: X_resampled, y_resampled
    def resample_smote(self, columns_training):
        X, y = self.get_records_and_labels(columns_training)
        sm = SMOTE()
        return sm.fit_sample(X, y)

    @staticmethod
    def get_records_and_labels(df, columns):
        return df[columns].values, df["simple_journal"].values

    # Output: dataframe with selected features
    def get_selected_features(self, feature_list):
        return self.df[feature_list]

    # def split_to_train_test(self, perc):
    #     number_of_records = self.df.shape[0]
    #     train_set = self.df.iloc[:int(number_of_records * perc)]
    #     test_set = self.df.iloc[int(number_of_records * perc):]
    #     return train_set, test_set

    # Output format: {"US": 112, ...}
    def total_per_country(self):
        result = self.df["issuercountrycode"].value_counts().to_dict()
        #result.to_csv(path=path,index_label=["issuercountrycode","transaction_count"],index=True)
        return result

    # Output: dictionary
    def total_per_cardtype(self):
        result = self.df["txvariantcode"].value_counts().to_dict()
        return result

    def total_per_cardid(self):
        result = self.df["card_id"].value_counts().to_dict()
        return result

    # Output: dataframe that contains only the "category"
    def filter_records(self, category):
        result = self.df.loc[self.df["simple_journal"] == category]
        return result


    def plot_data_balance(self):
        objects = tuple(self.df["simple_journal"].unique())
        y_pos = np.arange(len(objects))
        num_chargeback = self.df.loc[self.df["simple_journal"] == "Chargeback"].shape[0]
        num_refused = self.df.loc[self.df["simple_journal"] == "Refused"].shape[0]
        num_settled = (self.df.loc[self.df["simple_journal"] == "Settled"]).shape[0]
        number_of_records = [num_chargeback, num_refused, num_settled]
        plt.bar(y_pos, number_of_records, align='center', alpha=0.4, color='red')
        plt.xticks(y_pos, objects)
        plt.ylabel('Number of Transactions`')
        plt.title('Balance of the Data')
        plt.show()

    @staticmethod
    # The minority class gets oversampled to balance with the majority class
    # Output format: X_resampled, y_resampled
    def resample_smote2(X, y):
        sm = SMOTE()
        return sm.fit_sample(X, y)

    @staticmethod
    def string_to_timestamp(date_string):  # convert time string to float value
        time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return time.mktime(time_stamp)

    @staticmethod
    def get_percentage_of_frauds(trans_dict, chargebacks_dict):
        normalized_fpc = {}
        for key in trans_dict:
            if key in chargebacks_dict:
                normalized_fpc[key] = chargebacks_dict[key] / trans_dict[key]
        return normalized_fpc

    @staticmethod
    def plot_dictionary_sorted(D, figure_title):
        sorted_D = sorted(D.items(), key = operator.itemgetter(1))
        values = [x[1] for x in sorted_D]
        keys = [x[0] for x in sorted_D]
        plt.bar(range(len(sorted_D)), values, align='center',alpha=0.4, color = 'red')
        plt.xticks(range(len(sorted_D)), keys)
        plt.title(figure_title)
        plt.show()

    @staticmethod
    def get_plots():
        # barplot with number of settled, chargebacks and refused transactions
        trans_obj = Fraud()
        trans_obj.plot_data_balance()
        # get transactions per country (dictionary)
        trans_per_country = trans_obj.total_per_country()
        # print("Number of countries in the dataset: %d" % len(trans_per_country))

        # plot normalized fraud per countries
        # get fraud transactions per country (dictionary)
        chargebacks_obj = Fraud()
        chargebacks_obj.df = chargebacks_obj.filter_records("Chargeback")
        fraud_per_country = chargebacks_obj.total_per_country()
        print("Number of countries with frauds in the dataset: %d" % len(fraud_per_country))
        normalized_fpc = Fraud.get_percentage_of_frauds(trans_per_country, fraud_per_country)
        Fraud.plot_dictionary_sorted(normalized_fpc, "Normalized number of Frauds per country")

        # plot normalized cardType
        # get transactions per card type (dictionary)
        trans_per_card_type = trans_obj.total_per_cardtype()
        # get fraud transactions per card type (dictionary)
        fraud_per_cardtype = chargebacks_obj.total_per_cardtype()
        normalized_cardtype = Fraud.get_percentage_of_frauds(trans_per_card_type, fraud_per_cardtype)
        Fraud.plot_dictionary_sorted(normalized_cardtype, "Normalized  number of frauds per cardtype")

        # boxplots for settled and fraud amounts
        # get settled transactions
        settled_obj = Fraud()
        settled_obj.df = settled_obj.filter_records("Settled")

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
        axes = settled_obj.df.boxplot(column="amount", ax=ax1, sym='', showfliers=True)
        ax1.margins(y=0.05)
        ax1.set_title("settled")

        box_chargeback = chargebacks_obj.df.boxplot(column="amount", ax=ax2, sym='', showfliers=True)
        ax2.margins(y=0.05)
        ax2.set_title("Chargeback")
        plt.show()

    @staticmethod
    def knn(features_train, labels_train, variables):
        k = variables["k"]
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(features_train, labels_train)
        return knn_classifier

    @staticmethod
    def get_classifier(name_classifier, features_train, labels_train, variables):
        classifiers = {"knn": Fraud.knn}
        return classifiers[name_classifier](features_train, labels_train, variables)

    @staticmethod
    def write_to_file(classifier_name, variables, results):
        result = "Variables: %s\nTP: %s\nFP: %s\nFN: %s\nTN: %s\nAccuracy: %s\nPrecision: %s\nRecall: %s\nAUC: %s\nF1: %s\n\n" % (
            str(variables), results["TP"], results["FP"], results["FN"], results["TN"],
            results["accuracy"],results["precision"],results["recall"], results["auc"], results["f1"]
        )
        with open(os.path.join("..", "results", classifier_name), "a") as myfile:
            myfile.write(result)
        myfile.close()

    @staticmethod
    def evaluate(feature_vector, labels, classifier_name, variables, use_smote):
        n_splits = 2

        unique, counts = np.unique(labels, return_counts=True)
        print("Total label distribution", dict(zip(unique, counts)))

        convert_zero_one = lambda x: map(lambda y: 0 if y == "Settled" else 1, x)
        labels = np.array(convert_zero_one(labels))
        k_fold = StratifiedKFold(n_splits=n_splits)
        real_labels = []
        predicted_labels = []
        probability_labels = []
        print("Applying %s-fold crossvalidation with %s predictor, variables %s and smote=%s" % (n_splits, classifier_name,str(variables), str(use_smote)))
        for train, test in k_fold.split(feature_vector,labels):
            if use_smote:
                features_train, labels_train = Fraud.resample_smote2(feature_vector[train], labels[train])
            else:
                features_train, labels_train = feature_vector[train], labels[train]
            features_test, labels_test = feature_vector[test], labels[test]

            # Print distribution of the split
            unique, counts = np.unique(labels_test, return_counts=True)
            print("Current fold distribution", dict(zip(unique, counts)))

            # Build classifier using training set
            classifier = Fraud.get_classifier(classifier_name, features_train, labels_train, variables)

            # Save the labels
            real_labels.extend(labels_test)
            predicted_labels.extend(classifier.predict(features_test))
            probability_labels.extend(classifier.predict_proba(features_test))

        print("\tFinished")
        print("Retrieving results.")
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        accuracy = accuracy_score(real_labels, predicted_labels)
        f1 = f1_score(real_labels, predicted_labels)
        precision = average_precision_score(real_labels, predicted_labels)
        recall = recall_score(real_labels, predicted_labels)
        get_probability_labels = lambda container: map(lambda index: container[index][predicted_labels[index]], range(len(container)))
        auc = roc_auc_score(real_labels, get_probability_labels(probability_labels))

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
                             "precision":precision, "recall": recall, "auc": auc}
        print("Result:%s" % str(resulting_metrics))
        print("\tFinished!")

        print("Writing to file")
        Fraud.write_to_file(classifier_name, variables, resulting_metrics)
        print("\tFinished!")

        return resulting_metrics

    def run(self):
        print("Creating dummy variables.")
        filtered_df = self.get_selected_features(selected_features)
        dummies_df = pd.get_dummies(filtered_df, columns=["txvariantcode", "issuercountrycode", "shopperinteraction"])
        dummies_df = dummies_df.dropna(axis=0, how='any')
        print("\tFinished!!")

        print("Applying PCA.")
        features_without_labels = list(dummies_df)
        features_without_labels.remove(label)
        features_list, labels_list = self.get_records_and_labels(dummies_df, features_without_labels)
        pca = decomposition.PCA(n_components=30)
        pca_features = pca.fit_transform(features_list)
        print("\tFinished!!")

        print("Building classifier and apply SMOTE")
        Fraud.evaluate(pca_features, labels_list, "knn", {"k":3}, use_smote=True)
        print("\tFinished!!")


# LET THE MAGIC BEGIN

#### REMEMBER ############################################
# issuercountrycode: needs different columns per country {1,0}
# txvariantcode: same as above
# shopperinteraction: {Ecom, ContAuth, POS} needs different columns
# cardverificationcodesupplied: {True, False} which for python => {1,0} so it's ok
# amount: numerical, can be used as it is
# cvcresponsecode: binary, can be used as it is
#### STOP REMEMBERING ############################################

# Initialization of dataframe object (transactions)
trans_obj = Fraud()

# Selection of specified features (returns dataframe)
#trans_sel_features = Fraud()
#trans_sel_features.df = trans_obj.get_selected_features(selected_features)

#print trans_sel_features.df.shape

# convertion of features for SMOTE
#trans_for_SMT = Fraud()
#trans_for_SMT.df = pd.get_dummies(trans_sel_features.df, columns=["txvariantcode","issuercountrycode", "shopperinteraction"])


# sort dataset by date
#trans_for_SMT.df['creationdate'] =pd.to_datetime(trans_for_SMT.df.creationdate)
#trans_for_SMT.df.sort_values(by="creationdate")

# remove rows with missed values
#trans_for_SMT.df = trans_for_SMT.df.dropna(axis=0, how='any')

# split dataset to train and test set
# IMPORTANT: smote only to training set !!
# Take into account date ordering


# features_without_labels = list(trans_for_SMT.df)
# features_without_labels.remove(label)
# features_list, labels_list = trans_for_SMT.get_records_and_labels(features_without_labels)

#X_train, X_test, y_train, y_test = train_test_split(features_list, labels_list, test_size=0.55,random_state=42)

#train_set, test_set = trans_for_SMT.split_to_train_test(0.4)

# convert train_set and test_set to Fraud object
#train_obj = Fraud()
#train_obj.df = train_set

#test_obj = Fraud()
#test_obj.df = test_set

# PCA
#print X_train
#pca = decomposition.PCA(n_components=15)
#pca_features = pca.fit_transform(features_list)
#X_test = pca.transform(X_test)


# SMOTE
# create training features list
#training_features = list(train_set)
#print training_features
# training features => remove label, and creation time
#training_features.remove(label)

# sampling
#print X_train
#print y_train

#features_train, labels_train = Fraud.resample_smote2(X_train, y_train)

# KNN classifier
#neigh = KNeighborsClassifier(n_neighbors=4)
#neigh.fit(X_train, y_train)
#print neigh.score(X_test, y_test)
#y_predict = Fraud.evaluate_knn(pca_features, labels_list, 3)
    #neigh.predict(X_test)


#Fraud.knn_classifier(features_train, labels_train, features_test, labels_test)



#filter the dataframe per simple_journal category
# chargebacks_obj = Fraud()
# chargebacks_obj.df = chargebacks_obj.filter_records("Chargeback")
# settled_obj = Fraud()
# settled_obj.df = settled_obj.filter_records("Settled")
#refused_obj = Fraud()
#refused_obj.df = refused_obj.filter_records("Refused")


#get plots
#Fraud.get_plots()


# NOTES
# simple_journal = {Settled, Chargeback} We have removed "Refused"

# print the different values of 'simple_journal'
#print(trans_obj.df["simple_journal"].unique())

#df.to_pickle("augmented_dataframe")
#print(df.shape)