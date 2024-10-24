# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Copyright (c) 2022, Megagon Labs, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from packaging import version

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import transformers
from transformers import BertPreTrainedModel, BertConfig

if version.parse(transformers.__version__) < version.parse("4.1.1"):
    from transformers.modeling_bert import BertEmbeddings, BertEncoder
else:
    from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder


'''
The code used in this project includes several key components such as Column Type Annotations (CTA), 
Column Property Annotations (CPA), and Column Qualifier Annotations (CQA), which are crucial for the 
experiments and analyses. However, the rest of the code is not included here, as it is not intended 
for public release at this time. This is to safeguard ongoing research and maintain data integrity. 
Access to the code can be granted for academic or collaborative purposes upon request.
'''


if __name__ == "__main__":
    from transformers import BertTokenizer
    shortcut_name = "bert-base-multilingual-cased"
    model_config = BertConfig.from_pretrained(shortcut_name)
    model = BertForMultiOutputClassification(model_config)
