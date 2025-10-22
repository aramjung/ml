import os
import time
import pandas as pd
import zipfile
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def find_train_file():
    # Simplified: assume the archive is named 'train.tsv.zip'
    candidates = [
        os.path.join('data', 'train.tsv.zip'),
        os.path.join('..', 'data', 'train.tsv.zip')
    ]
    # Check Kaggle input folder for an archive named train.tsv.zip
    kaggle_root = '/kaggle/input'
    if os.path.exists(kaggle_root):
        for root, dirs, files in os.walk(kaggle_root):
            for f in files:
                if f.lower() == 'train.tsv.zip' or f.lower().endswith('/train.tsv.zip') or f.lower().endswith('train.tsv.zip'):
                    return os.path.join(root, f)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not locate 'train.tsv.zip' - expected at data/ or under /kaggle/input/")


def load_data():
    path = find_train_file()
    # Handle zipped files containing the train data (e.g., train.tsv.zip)
    if path.lower().endswith('.zip'):
        # open the zip and prefer an inner file named 'train.tsv'
        with zipfile.ZipFile(path, 'r') as z:
            names = z.namelist()
            candidate = None
            for name in names:
                b = os.path.basename(name).lower()
                if b == 'train.tsv':
                    candidate = name
                    break
            # fallback: pick first tsv/csv containing 'train' or the first member
            if candidate is None:
                for name in names:
                    b = os.path.basename(name).lower()
                    if 'train' in b and (b.endswith('.tsv') or b.endswith('.csv')):
                        candidate = name
                        break
            if candidate is None and len(names) == 1:
                candidate = names[0]
            if candidate is None:
                raise FileNotFoundError(f"No train .tsv/.csv found inside zip {path} (members: {names})")
            with z.open(candidate) as fh:
                if candidate.lower().endswith('.tsv'):
                    df = pd.read_csv(fh, sep='\t')
                else:
                    df = pd.read_csv(fh)
    else:
        # TSV is expected for this dataset; allow csv with \t separator as fallback
        if path.lower().endswith('.tsv'):
            df = pd.read_csv(path, sep='\t')
        else:
            # try to read as csv but allow tab separator if necessary
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, sep='\t')
    return df


df = load_data()
df.head()

# create dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # encodings can be a dict of lists (python) or a dict of tensors (return_tensors='pt')
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        # if values are tensors (return_tensors='pt'), indexing returns per-example tensors
        first_val = next(iter(self.encodings.values()))
        if isinstance(first_val, torch.Tensor):
            for k, v in self.encodings.items():
                item[k] = v[idx]
        else:
            for k, v in self.encodings.items():
                item[k] = torch.tensor(v[idx])
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

print('dataset class defined')

# Basic sanity fixes
if 'Phrase' not in df.columns or 'Sentiment' not in df.columns:
    raise KeyError('Expected columns Phrase and Sentiment in the training file')

df['Phrase'] = df['Phrase'].fillna('').astype(str)
if not pd.api.types.is_integer_dtype(df['Sentiment']):
    df['Sentiment'] = df['Sentiment'].astype(int)
    
# train / validation split - split by SentenceId to avoid data leakage when present
if 'SentenceId' in df.columns:
    train_ids, val_ids = train_test_split(df['SentenceId'].unique(), test_size=0.2, random_state=42)
    train_df = df[df['SentenceId'].isin(train_ids)]
    val_df = df[df['SentenceId'].isin(val_ids)]
else:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

train_texts = train_df['Phrase'].tolist()
train_labels = train_df['Sentiment'].tolist()
val_texts = val_df['Phrase'].tolist()
val_labels = val_df['Sentiment'].tolist()

# tokenize texts using the fast tokenizer and return PyTorch tensors for efficiency
try:
    tokenizer = BertTokenizerFast.from_pretrained('/kaggle/input/bert-downloaded')
except Exception as e:
    # Some environments (offline, restricted network, or older HF repos) may fail to fetch
    # the fast tokenizer. Fall back to the (slower) Python tokenizer with a clear message.
    print(f"Warning: failed to load fast tokenizer: {e}\nFalling back to BertTokenizer (slow). If you're on Kaggle, enable internet or provide the tokenizer in the working directory to use the fast tokenizer.")
    try:
        tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bert-downloaded')
    except Exception as e2:
        raise RuntimeError("Failed to load any tokenizer. Enable internet or supply tokenizer files locally.") from e2
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=64, return_tensors='pt')

print('tokenization complete')

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
# create dataloaders
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# DataLoader settings: smaller batch on CPU, sensible num_workers on Kaggle (Linux)
is_cuda = torch.cuda.is_available()
default_batch = 8 if is_cuda else 4
num_workers = 2 if os.name != 'nt' else 0
train_loader = DataLoader(train_dataset, batch_size=default_batch, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=default_batch, shuffle=False, num_workers=num_workers)

# load model
num_labels = int(df['Sentiment'].nunique())
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

print('Dataloaders created.  Model is loaded.')
# optimizer and device setup
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# training loop
for epoch in range(3):
    st = time.time()
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_loader)
    t_elapsed = time.time() - st
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, time: {t_elapsed:.1f}s")

    # validation
    model.eval()
    val_labels_list = []
    val_preds_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_labels_list.extend(labels.cpu().numpy())
            val_preds_list.extend(preds.cpu().numpy())
    val_accuracy = accuracy_score(val_labels_list, val_preds_list)
    print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")
    print(classification_report(val_labels_list, val_preds_list))

# save model to a sensible location (Kaggle kernels write to /kaggle/working)
output_dir = os.path.join('/kaggle/working', 'model') if os.path.exists('/kaggle/working') else os.path.join(os.getcwd(), 'model')
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved model and tokenizer to {output_dir}")
