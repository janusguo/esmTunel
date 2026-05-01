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

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Ensure labels match model dtype
            model_dtype = next(model.parameters()).dtype
            labels = batch['label'].to(device).unsqueeze(1).to(model_dtype)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Apply sigmoid for metrics
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
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
        'loss': total_loss / len(data_loader),
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

import argparse

def train():
    parser = argparse.ArgumentParser(description="Fine-tune ESM model for protein allergenicity prediction")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps per epoch")
    args = parser.parse_args()

    # Parameters
    model_name = args.model_name
    data_dir = "./data"
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print(f"Data files not found in {data_dir}. Please ensure bio_researcher has prepared train.csv, val.csv, and test.csv.")
        return

    # Load Datasets
    print("Loading datasets...")
    train_dataset = ProteinDataset(train_path, tokenizer_name=model_name)
    val_dataset = ProteinDataset(val_path, tokenizer_name=model_name)
    test_dataset = ProteinDataset(test_path, tokenizer_name=model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize Model
    model = ESMAllergenicityModel(model_name=model_name).to(device)
    
    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    best_val_loss = float('inf')
    patience = 3
    counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_auc': []
    }

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(train_loop):
            if args.max_steps > 0 and i >= args.max_steps:
                break
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Ensure labels match model dtype
            model_dtype = next(model.parameters()).dtype
            labels = batch['label'].to(device).unsqueeze(1).to(model_dtype)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(metrics['loss'])
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1'])
        history['val_auc'].append(metrics['auc'])
        
        import json
        with open('training_history.json', 'w') as f:
            json.dump(history, f)
        
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
    
    # Test Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt"))
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    with open('test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)
    
    print("Results saved to test_metrics.json")

if __name__ == "__main__":
    train()
