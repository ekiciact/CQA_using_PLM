import pandas as pd
import os

# List of table files
tables_with_1_example = ['table_93.csv', 'table_553.csv', 'table_572.csv', 'table_929.csv', 'table_945.csv', 'table_957.csv', 'table_1367.csv', 'table_1444.csv', 'table_1521.csv', 'table_1589.csv', 'table_1968.csv', 'table_2044.csv', 'table_2074.csv', 'table_2228.csv', 'table_2366.csv', 'table_2580.csv']


# Directories
input_directory = 'train'
output_directory = 'split_tables'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)


def split_file(file_name):
    # Construct the full file path
    input_path = os.path.join(input_directory, file_name)

    # Read the CSV file
    df = pd.read_csv(input_path)

    # Shuffle the DataFrame rows
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # Split the DataFrame into two halves
    half = len(df_shuffled) // 2
    df_half1 = df_shuffled.iloc[:half, :]
    df_half2 = df_shuffled.iloc[half:, :]

    # Save the new CSV files
    file_base, file_extension = os.path.splitext(file_name)
    df_half1.to_csv(os.path.join(output_directory, f"{file_base}_half_1.csv"), index=False)
    df_half2.to_csv(os.path.join(output_directory, f"{file_base}_half_2.csv"), index=False)


# Process each file in the list
for table_file in tables_with_1_example:
    split_file(table_file)
