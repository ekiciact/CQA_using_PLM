import os
import pickle
import random
import time
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import transformers
from transformers import AutoTokenizer
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
# tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

def load(filename):
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data

def save(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

'''
The code used in this project includes several key components such as Column Type Annotations (CTA), 
Column Property Annotations (CPA), and Column Qualifier Annotations (CQA), which are crucial for the 
experiments and analyses. However, the rest of the code is not included here, as it is not intended 
for public release at this time. This is to safeguard ongoing research and maintain data integrity. 
Access to the code can be granted for academic or collaborative purposes upon request.
'''

if __name__ == "__main__":
    # test_cta()
    # test_cpa()
    test_cqa()