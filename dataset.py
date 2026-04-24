import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class ProteinDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="facebook/esm2_t33_650M_UR50D", max_length=512):
        print(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]['sequence']
        label = self.df.iloc[idx]['label']
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }
