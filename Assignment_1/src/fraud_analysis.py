#!/usr/bin/python3
import pandas as pd
import numpy as np
import time


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

    # Output: dataframe that contains only the "Chargebacks"
    def filter_chargeback_records(self):
        result = self.df.loc[self.df["simple_journal"] == "Chargeback"]
        return result

    # Output: dataframe that contains only the "Settled"
    def filter_settled_records(self):
        result = self.df.loc[self.df["simple_journal"] == "Settled"]
        return result

    def filter_refused_records(self):
        result = self.df.loc[self.df["simple_journal"] == "Refused"]
        return result

    @staticmethod
    def string_to_timestamp(date_string):  # convert time string to float value
        time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return time.mktime(time_stamp)



# Initialization of dataframe object
fraud_obj = Fraud()
print(fraud_obj.df.shape)
# print fraud per country
#print(fraud_obj.total_per_country())

# print the different values of 'simple_journal'
print(fraud_obj.df["simple_journal"].unique())


chargebacks_df = fraud_obj.filter_chargeback_records()
#print(chargebacks_df)
settled_df = fraud_obj.filter_settled_records()

refused_df = fraud_obj.filter_refused_records()

print(chargebacks_df.shape)

print(settled_df.shape)

print (refused_df.shape)


# NOTES
# simple_journal = {Settled, Chargeback} We have removed "Refused"