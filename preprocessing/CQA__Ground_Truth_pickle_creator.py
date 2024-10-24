import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

# Read the data from the file 'wikidata_qualifier_labels.csv'
qualifier_labels_df = pd.read_csv('wikidata_qualifier_labels.csv')

label2idx = {}  # This dictionary maps labels to indices
idx2label = {}  # This dictionary maps indices to labels
qualifiers = {}

# Loop over the rows of train_df
for index, row in qualifier_labels_df.iterrows():
    # Extract the values of the columns
    property = row['Property']
    label = row['Label']

    label2idx[label] = index  # Assign the index to the label
    idx2label[index] = label  # Assign the label to the index
    qualifiers[property] = index

# Read the 'train.csv' file using pandas
train_df = pd.read_csv('train.csv')
filtered_tables = []
# Filter out the specified tables (non_zero tables)
# non_zero_tables = ['table_8.csv', 'table_42.csv', 'table_52.csv', 'table_59.csv', 'table_143.csv', 'table_202.csv',
#                    'table_253.csv', 'table_450.csv', 'table_493.csv', 'table_510.csv', 'table_513.csv', 'table_586.csv',
#                    'table_619.csv', 'table_750.csv', 'table_802.csv', 'table_846.csv', 'table_857.csv', 'table_875.csv',
#                    'table_945.csv', 'table_1020.csv', 'table_1226.csv', 'table_1228.csv', 'table_1262.csv', 'table_1314.csv',
#                    'table_1414.csv', 'table_1440.csv', 'table_1452.csv', 'table_1651.csv', 'table_1660.csv', 'table_1826.csv',
#                    'table_1900.csv', 'table_1931.csv', 'table_1968.csv', 'table_2055.csv', 'table_2108.csv', 'table_2116.csv',
#                    'table_2141.csv', 'table_2187.csv', 'table_2365.csv', 'table_2371.csv']
# filtered_tables.extend(non_zero_tables)
# #
# # # #
# # # # Filter out the specified tables (1 example tables)
one_example_tables = ['table_93.csv', 'table_553.csv', 'table_572.csv', 'table_929.csv', 'table_945.csv', 'table_957.csv', 'table_1367.csv', 'table_1444.csv', 'table_1521.csv', 'table_1589.csv', 'table_1968.csv', 'table_2044.csv', 'table_2074.csv', 'table_2228.csv', 'table_2366.csv', 'table_2580.csv']

filtered_tables.extend(one_example_tables)

train_df = train_df[~train_df['File Name'].isin(filtered_tables)]

# Create an empty dictionary to store the data for the CQA pickle file
cqa_dict = {}

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

    # Check if the file_name is already in the cqa_dict
    if file_name not in cqa_dict:
        # Create a new entry for it
        cqa_dict[file_name] = {
            'col_idx': [],
            'label': [],
        }

    # Append a tuple of (subject_col, object_col, qualifier_col) to the 'col_idx' list
    cqa_dict[file_name]['col_idx'].append(sorted([subject_col, object_col, qualifier_col]))

    # Append a tuple of (property_id, qualifier_id) to the 'label' list
    cqa_dict[file_name]['label'].append(qualifiers[qualifier_id])

# Count examples for each label
label_counts = Counter()
for data in cqa_dict.values():
    label_counts.update(data['label'])

# Sort labels by the number of examples (ascending) and table names by descending order
sorted_label_counts = sorted(
    label_counts.items(),
    key=lambda x: (
        x[1],
        -int(max(
            [int(f.split('_')[1].split('.')[0]) for f in train_df[train_df['Property Qualifier ID'] == idx2label[x[0]]]['File Name']]
            if train_df[train_df['Property Qualifier ID'] == idx2label[x[0]]]['File Name'].tolist() else [0] # Add a default value to avoid empty sequence
        ))
    )
)

# Print examples for each label and collect one-example tables
one_examples = 0
one_example_tables = {}  # Dictionary to store labels and their one-example tables
one_example_table_names = []  # List to store the names of tables with one-example labels
print("Examples for each label (sorted by number of examples and table name):")
for label, count in sorted_label_counts:
    if count == 1:
        one_examples += 1
        tables = [key for key, value in cqa_dict.items() if label in value['label']]
        one_example_tables[label] = tables
        one_example_table_names.extend(tables)
    print(f"Label {label} ({idx2label[label]}): {count} examples")

# Print the table names for one-example labels
print("\nTable names for one-example labels:")
for label, tables in one_example_tables.items():
    print(f"Label {label} ({idx2label[label]}): {tables}")

# Print the list of one-example table names
print("\nList of table names with one-example labels:")
print(one_example_table_names)

# Split the cqa_dict into train and validation sets based on the label counts
train_keys = []
val_keys = []

for label, count in label_counts.items():
    keys_with_label = [key for key, value in cqa_dict.items() if label in value['label']]
    if count >= 20:
        train_size = 0.85
    elif count <= 1:
        train_keys.extend(keys_with_label)
        continue
    elif count <= 3:
        train_size = 0.67
    elif count <= 4:
        train_size = 0.75
    elif count <= 10:
        train_size = 0.80
    else:
        train_size = 0.80  # default to 0.80 if not specified

    train_keys_label, val_keys_label = train_test_split(keys_with_label, train_size=train_size, random_state=42)
    train_keys.extend(train_keys_label)
    val_keys.extend(val_keys_label)

# Randomize the train and validation keys
np.random.shuffle(train_keys)
np.random.shuffle(val_keys)

cqa_train = {key: cqa_dict[key] for key in train_keys}  # create the train dictionary
cqa_val = {key: cqa_dict[key] for key in val_keys}  # create the validation dictionary

# Calculate and print the final train/validation ratio
total_train = len(train_keys)
total_val = len(val_keys)
train_val_ratio = total_train / (total_train + total_val) if (total_train + total_val) > 0 else 0

# Create the final CQA pickle file data
cqa_data = {
    'train': cqa_train,
    'validation': cqa_val,
    'label2idx': label2idx,
    'idx2label': idx2label
}

# Print out important information
print("\nImportant Information:")
print(f"Number of 1 examples: {one_examples}")
print(f"Total number of examples: {len(train_df)}")
print(f"Number of training examples: {total_train}")
print(f"Number of validation examples: {total_val}")
print(f"Final train/validation ratio: {train_val_ratio:.2f}")

# Save the cqa_data to a pickle file
with open('CQA-DBP.pkl', 'wb') as f:
    pickle.dump(cqa_data, f)
