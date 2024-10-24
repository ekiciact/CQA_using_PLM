from transformers import AutoTokenizer
import pickle
import pprint
import pandas as pd

file = 'cqa_raw_data.pickle'

# Open the pickle file in read-binary mode
with open(file, 'rb') as f:
    # Load the data from the pickle file
    input_dict = pickle.load(f)

# Choose a pre-trained tokenizer (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize the sentences in each list of lists
tokenized_inputs = {}
for key, row_lists in input_dict.items():

    tokenized_lists = []
    for row in row_lists:
        # Handle 'nan' values by replacing them with an empty string
        row = [str(cell) if pd.notna(cell) else "" for cell in row]

        tokenized_row = tokenizer(
            row,
            max_length=512,
            padding=False,
            truncation=False,
            is_split_into_words=False,
            return_tensors='np',
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        tokenized_lists.append(tokenized_row)

    tokenized_inputs[key] = tokenized_lists

# Format the output with 'input_ids' values as lists
formatted_output = {}
for key, tokenized_lists in tokenized_inputs.items():
    formatted_output[key] = []
    for tokenized_row in tokenized_lists:
        formatted_row_input_ids = []
        for i in range(len(tokenized_row['input_ids'])):
            formatted_row_input_ids.append(tokenized_row['input_ids'][i].tolist())  # Convert tensor to list
        formatted_output[key].append(formatted_row_input_ids)

# Print the formatted output
print("Formatted Output:")
# pprint.pprint(formatted_output)
print(formatted_output.get('table_2196.csv', 'Table not found'))

# Save the tokenized data to a pickle file
with open('cqa.pkl', 'wb') as f:
    pickle.dump(formatted_output, f)
