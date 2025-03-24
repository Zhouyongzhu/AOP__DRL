import Getmetrics
from sklearn.model_selection import train_test_split
import datasets
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer
from transformers import BertForMaskedLM, BertTokenizer, pipeline, BertForSequenceClassification
import os
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import Getmetrics
import getDataset
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
def getDataset(modelpath='', tkpath='', datapath='', max_len=512):
    do_lower = False
    tokenizer = AutoTokenizer.from_pretrained(tkpath, do_lower_case=do_lower)

    print('Tokenizer have been loaded.')
    dataset = datasets.load_dataset("csv", cache_dir='/home/ubuntu/zhouyongzhu/data/cache',
                                    data_files=datapath)
    dataset = dataset.shuffle(seed=702)
    print('Dataset have been loaded.')
    tokenized_dataset = dataset.map(lambda x: tokenizer(
        ' '.join(x["sequence"]), max_length=max_len,
        padding="max_length", truncation=True))
    return tokenized_dataset['train']
