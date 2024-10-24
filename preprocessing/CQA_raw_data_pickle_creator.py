
# Import the necessary libraries
import pyarrow
import csv
import pprint
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


# Read the 'train.csv' file using pandas
train_df = pd.read_csv('train.csv')


# Create an empty dictionary to store the data for the CQA pickle file
cqa_raw_data = {}

# Loop over the rows of train_df
for index, row in train_df.iterrows():
  # Extract the values of the columns
  file_name = row['File Name']
  subject_col = row['Subject Column']
  object_col = row['Object Column']
  qualifier_col = row['Qualifier Column']
  property_id = row['Property ID']
  property_label = row['Property Label']
  qualifier_id = row['Property Qualifier ID']
  qualifier_label = row['Property Qualifier Label']

  # Read the file with the file_name using pandas
  table_df = pd.read_csv(f'train/{file_name}', encoding="utf-8")

  # if file_name == 'table_2638.csv':
  #   print(table_df)

  # initialize an empty list
  list_of_lists = []

  # iterate over the columns and add each column as a list to the list_of_lists
  if file_name == "table_1444.csv" or file_name == "table_1444_half_2.csv" or file_name == "table_1444_half_1.csv":
    print("1")
  for index, row in table_df.iterrows():
    list_of_lists.append(row.values.tolist())
    # if file_name == "table_8.csv":
    #   print(row)

  # if index == 1:
  #   break

  # if file_name == "table_1451.csv":
  #   print (table_df)

  # Check if the file_name is already in the cqa_dict
  if file_name not in cqa_raw_data:
    # Create a new entry for it
    cqa_raw_data[file_name] = list_of_lists

# pprint.pprint(cqa_raw_data)

# Save the cqa_data to a pickle file
with open('cqa_raw_data.pickle', 'wb') as f:
  pickle.dump(cqa_raw_data, f)
