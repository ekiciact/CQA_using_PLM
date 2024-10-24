import subprocess
import argparse
import json
import math
import os
import random
from time import time
import pickle
import wandb
import datetime
import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


from pt_dataset import (
    collate_fn,
    collate_test_fn,
    CTADataset,
    CTATestDataset,
    CPADataset,
    CPATestDataset,
    CQADataset,
)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import f1_score_multilabel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()


parser.add_argument(
    "--run_mode",
    choices=["sequential", "round_robin"],
    default="round_robin",
    help="Choose 'sequential' to run each task for the full number of epochs before moving to the next task. Choose 'round_robin' to run each task one epoch at a time in a round-robin fashion."
)
parser.add_argument(
    "--shortcut_name",
    default="bert-base-multilingual-cased",
    # default="xlm-roberta-base",
    type=str,
    help="Huggingface model shortcut name ",
)

'''
The code used in this project includes several key components such as Column Type Annotations (CTA), 
Column Property Annotations (CPA), and Column Qualifier Annotations (CQA), which are crucial for the 
experiments and analyses. However, the rest of the code is not included here, as it is not intended 
for public release at this time. This is to safeguard ongoing research and maintain data integrity. 
Access to the code can be granted for academic or collaborative purposes upon request.
'''


if __name__ == "__main__":
    train()

    # Analyze errors after training for each task and epoch
    task_names = "_".join(args.tasks)
    if "CQA-DBP" in args.tasks:
        filename = f'{task_names}_{run_option}_predictions_and_labels.pkl'
    else:
        filename = f'{task_names}_predictions_and_labels.pkl'

    predictions_and_labels_path = os.path.join(model_save_path, filename)
    analyze_errors(predictions_and_labels_path, num_classes=len(num_classes_list[0]))
