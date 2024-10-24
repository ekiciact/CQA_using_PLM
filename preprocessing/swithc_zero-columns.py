import pandas as pd
import os

# Load the train.csv file
train_df = pd.read_csv('train.csv')

# List of specified tables non_zero tables
tables = ['table_8.csv', 'table_42.csv', 'table_52.csv', 'table_59.csv', 'table_143.csv', 'table_202.csv',
          'table_253.csv', 'table_450.csv', 'table_493.csv', 'table_510.csv', 'table_513.csv', 'table_586.csv',
          'table_619.csv', 'table_750.csv', 'table_802.csv', 'table_846.csv', 'table_857.csv', 'table_875.csv',
          'table_945.csv', 'table_1020.csv', 'table_1226.csv', 'table_1228.csv', 'table_1262.csv', 'table_1314.csv',
          'table_1414.csv', 'table_1440.csv', 'table_1452.csv', 'table_1651.csv', 'table_1660.csv', 'table_1826.csv',
          'table_1900.csv', 'table_1931.csv', 'table_1968.csv', 'table_2055.csv', 'table_2108.csv', 'table_2116.csv',
          'table_2141.csv', 'table_2187.csv', 'table_2365.csv', 'table_2371.csv']

# Filter the dataframe to include only the specified tables and extract the 'Subject Column'
filtered_df = train_df[train_df['File Name'].isin(tables)][['File Name', 'Subject Column']]

# Step 1: Update the 'Subject Column' in train.csv to 0 for the specified tables
train_df.loc[train_df['File Name'].isin(tables), 'Subject Column'] = 0
# train_df.to_csv('updated_train.csv', index=False)
train_df.to_csv('train.csv', index=False)

# Step 2: Update the data in the source directory and save to new directory
source_directory = 'train'
destination_directory = 'updated_tables'

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

for table in tables:
    file_path = os.path.join(source_directory, table)
    if os.path.exists(file_path):
        table_df = pd.read_csv(file_path)
        subject_col_index = filtered_df[filtered_df['File Name'] == table]['Subject Column'].values[0]

        # Convert the entire DataFrame to string type
        table_df = table_df.astype(str)

        # Swap the columns row by row
        for i in range(len(table_df)):
            temp = table_df.iloc[i, 0]
            table_df.iloc[i, 0] = table_df.iloc[i, subject_col_index]
            table_df.iloc[i, subject_col_index] = temp

        # Save the updated dataframe to the new directory
        new_file_path = os.path.join(destination_directory, table)
        table_df.to_csv(new_file_path, index=False)

print("Script executed successfully.")
