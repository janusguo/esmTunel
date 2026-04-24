import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import ESMAllergenicityModel
from dataset import ProteinDataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    preds_binary = (all_preds > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_binary, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, preds_binary)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    return {
        'loss': val_loss / len(val_loader),
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def train():
    # Parameters
    model_name = "facebook/esm2_t33_650M_UR50D"
    train_path = "/home/team/shared/data/train.csv"
    val_path = "/home/team/shared/data/val.csv"
    batch_size = 16
    lr = 1e-4
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Data not found. Please ensure bio_researcher has prepared train.csv and val.csv in /home/team/shared/data/")
        return

    # Load Datasets
    train_dataset = ProteinDataset(train_path, tokenizer_name=model_name)
    val_dataset = ProteinDataset(val_path, tokenizer_name=model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize Model
    model = ESMAllergenicityModel(model_name=model_name).to(device)
    
    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCELoss()

    # Early Stopping
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in train_loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

        # Early Stopping check
        if metrics['loss'] < best_val_loss:
            best_val_loss = metrics['loss']
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    train()
