import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('./bert-base-uncased')

train = pd.read_csv('data/train.tsv', sep='\t')
print(train.head())
print(train['Sentiment'].value_counts())
print(train.isnull().sum())


