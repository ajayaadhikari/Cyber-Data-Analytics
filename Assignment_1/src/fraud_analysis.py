from __future__ import division
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import operator


data_path = '../datasets/data_for_student_case.csv'
columns = ["txid", "bookingdate", "issuercountrycode", "txvariantcode", "card_issuer_identifier",
           "amount", "currencycode", "shoppercountrycode", "shopperinteraction", "simple_journal",
           "cardverificationcodesupplied", "cvcresponsecode", "creationdate", "accountcode", "mail_id",
           "ip_id", "card_id"]

class Fraud:
    def __init__(self):
        self.load_data()

    def load_data(self):
        # Read from csv
        self.df = pd.read_csv(data_path, skip_blank_lines=True)

        # Rename a column
        self.df.rename(columns={'bin': 'card_issuer_identifier'}, inplace=True)
        # Delete rows with null values for card_issuer_identifier
        self.df = self.df[pd.notnull(self.df.card_issuer_identifier)]
        self.df = self.df[self.df.simple_journal != "Refused"]

        # Change data types of some columns
        self.df["bookingdate"].apply(self.string_to_timestamp)
        self.df["card_issuer_identifier"].apply(float)
        self.df["amount"].apply(lambda x: float(x)/100)

    # The minority class gets oversampled to balance with the majority class
    # Output format: X_resampled, y_resampled
    def resample_smote(self, columns_training):
        X, y = self.get_records_and_labels(columns_training)
        sm = SMOTE()
        return sm.fit_sample(X, y)

    def get_records_and_labels(self, columns):
        return self.df[columns].values, self.df["simple_journal"].values

    # Output: dataframe with selected features
    def get_selected_features(self, feature_list):
        return self.df[feature_list]

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


# LET THE MAGIC BEGIN

selected_features = ["issuercountrycode", "txvariantcode", "amount", "shopperinteraction", "cardverificationcodesupplied",
                     "cvcresponsecode", "simple_journal", "creationdate"]

label = "simple_journal"
features_for_convertion = ["issuercountrycode", "txvariantcode", "shopperinteraction"]

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
trans_sel_features = Fraud()
trans_sel_features.df = trans_obj.get_selected_features(selected_features)

print trans_sel_features.df.shape
# convertion of features for SMOTE
trans_for_SMT = Fraud()
trans_for_SMT.df = pd.get_dummies(trans_sel_features.df, columns=["txvariantcode","issuercountrycode", "shopperinteraction"])


# sort dataset by date
trans_for_SMT.df['creationdate'] =pd.to_datetime(trans_for_SMT.df.creationdate)
trans_for_SMT.df.sort_values(by="creationdate")


# SMOTE
# create training features list
training_features = list(trans_for_SMT.df)
# training features => remove label, and creation time
training_features.remove("creationdate")
training_features.remove(label)

# remove rows with missed values
trans_for_SMT.df = trans_for_SMT.df.dropna(axis=0, how='any')
# sampling
x, y = trans_for_SMT.resample_smote(training_features)
print set(y)


#filter the dataframe per simple_journal category
chargebacks_obj = Fraud()
chargebacks_obj.df = chargebacks_obj.filter_records("Chargeback")
settled_obj = Fraud()
settled_obj.df = settled_obj.filter_records("Settled")
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