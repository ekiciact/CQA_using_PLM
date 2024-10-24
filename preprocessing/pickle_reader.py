# Import the pickle module
import pickle
import csv
import random


file = 'runOptions/CQA_run5/CQA-DBP.pkl'

# file = 'cqa_raw_data.pickle'

# file = 'cqa.pkl'
# Open the pickle file in read-binary mode
with open(file, 'rb') as f:
  # Load the data from the pickle file
  cqa_data = pickle.load(f)

# Print the data or do other operations
print(cqa_data)

validation = cqa_data['validation']
print(validation)

vl_true_list = []
vl_pred_list = []

# Initialize an empty dictionary
csv_data = {}

# Read the CSV file
with open('train_prediction_nofill.csv', mode='r') as file:
  csv_reader = csv.DictReader(file)

  # Loop through each row in the CSV
  for row in csv_reader:
    # Get the file name
    file_name = row.pop("File Name")

    # Use the file name as the key and the rest of the row as the value
    csv_data[file_name] = row

# Print the resulting dictionary
for file, details in csv_data.items():
  print(f"{file}: {details}")


label2idx = cqa_data['label2idx']

for key, value in validation.items(): # Key is the table name
  vl_true_list.append(value["label"][0])
  print(key, value["label"])

  if key not in csv_data:
    # Select a random element
    label_space = [idx for label, idx in label2idx.items() if value["label"][0] != idx]

    random_element = random.choice(label_space)
    print(random_element)

    vl_pred_list.append(random_element)
  else:
    label = label2idx[csv_data[key]['Property Qualifier Label']]
    vl_pred_list.append(label)

print(vl_true_list)
print(vl_pred_list)

