import json
import matplotlib.pyplot as plt
import os

def plot_metrics(history_file='training_history.json', output_dir='/home/team/shared/results/'):
    if not os.path.exists(history_file):
        print(f"History file {history_file} not found.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
        
    with open(history_file, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    # Plot Accuracy and F1
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    # Example usage, will be called after training
    plot_metrics('training_history.json', '/home/team/shared/results/')
