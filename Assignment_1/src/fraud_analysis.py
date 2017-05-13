from __future__ import division
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


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
        #self.df = self.df[self.df.simple_journal != "Refused"]

        # Change data types of some columns
        self.df["bookingdate"].apply(self.string_to_timestamp)
        self.df["card_issuer_identifier"].apply(float)
        self.df["amount"].apply(float)

    # Output format: {"US": 112, ...}
    def total_per_country(self):
        result = self.df["issuercountrycode"].value_counts().to_dict()
        #result.to_csv(path=path,index_label=["issuercountrycode","transaction_count"],index=True)
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
        plt.bar(y_pos, number_of_records, align='center', alpha=0.4)
        plt.xticks(y_pos, objects)
        plt.ylabel('Number of Transactions`')
        plt.title('Balance of the Data')
        plt.show()

    @staticmethod
    def string_to_timestamp(date_string):  # convert time string to float value
        time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return time.mktime(time_stamp)

    @staticmethod
    def get_percentage_of_frauds_per_country(trans_dict, chargebacks_dict):
        normalized_fpc = {}
        for key in trans_dict:
            if key in chargebacks_dict:
                normalized_fpc[key] = chargebacks_dict[key] / trans_dict[key]
        return normalized_fpc

# Initialization of dataframe object (transactions)
trans_obj = Fraud()
print(trans_obj.df.shape)

#filter the dataframe per simple_journal category
chargebacks_obj = Fraud()
chargebacks_obj.df = chargebacks_obj.filter_records("Chargeback")

settled_obj = Fraud()
settled_obj.df = settled_obj.filter_records("Settled")

#refused_df = fraud_obj.filter_records("Refused")


# get transactions per country (dictionary)
trans_per_country = trans_obj.total_per_country()
print("Number of countries in the dataset: %d" % len(trans_per_country))

# get fraud transactions per country (dictionary)
fraud_per_country = chargebacks_obj.total_per_country()
print("Number of countries with frauds in the dataset: %d" % len(fraud_per_country))


normalized_fpc = Fraud.get_percentage_of_frauds_per_country(trans_per_country, fraud_per_country)

print(normalized_fpc)


# print fraud per card id
#cardid_dict = trans_obj.total_per_cardid()
#plt.bar(list(cardid_dict.keys()), cardid_dict.values(), width=1.0, color='g')
#D = trans_per_country
#print(D.values())

#plt.bar(range(len(D)), D.values(), align='center')
#plt.xticks(range(len(D)), D.keys())

#plt.show()



# print the different values of 'simple_journal'
print(trans_obj.df["simple_journal"].unique())


#barplot with number of settled, chargebacks and refused transactions
trans_obj.plot_data_balance()


# NOTES
# simple_journal = {Settled, Chargeback} We have removed "Refused"