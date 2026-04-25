import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import os
import argparse
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ESMAllergenicityModel

def load_model(model_path, model_name="facebook/esm2_t33_650M_UR50D", device="cpu"):
    print(f"Loading model from {model_path}...")
    # Use ESMAllergenicityModel with eager attention
    model = ESMAllergenicityModel(model_name=model_name, attn_implementation="eager")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def get_attentions_and_saliency(model, tokenizer, sequence, device="cpu"):
    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Enable gradients for saliency
    model.zero_grad()
    
    # Access the ESM model inside the PEFT wrapper
    esm_model = model.esm.base_model.model
    # Re-load or ensure attention implementation is 'eager' to get attentions
    # We can't easily change it on the fly, so we should set it when loading
    
    # 1. Get Attention Maps
    with torch.no_grad():
        outputs = esm_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_attentions=True, 
            return_dict=True
        )
        attentions = outputs.attentions  # Tuple of (batch, heads, seq_len, seq_len)
    
    # 2. Saliency Mapping
    # We need to get gradients w.r.t embeddings
    input_embeddings = esm_model.embeddings.word_embeddings(input_ids).detach().requires_grad_(True)
    
    # Re-run forward pass with embeddings
    outputs_emb = esm_model(
        inputs_embeds=input_embeddings, 
        attention_mask=attention_mask,
        return_dict=True
    )
    cls_output = outputs_emb.last_hidden_state[:, 0, :]
    logits = model.classifier(cls_output)
    
    # Backward pass to get gradients
    logits.backward()
    
    # Saliency is the norm of the gradients
    saliency = input_embeddings.grad.data.abs().sum(dim=-1).squeeze()
    
    # Normalize saliency
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return attentions, saliency.cpu().numpy(), tokens, torch.sigmoid(logits).item()

def plot_attention_heatmap(attention, tokens, layer, head, output_path):
    # attention shape: (heads, seq_len, seq_len)
    attn = attention[head].detach().cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Attention Map - Layer {layer}, Head {head}")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_saliency(saliency, tokens, output_path, title="Residue Saliency Map", highlight_ranges=None):
    plt.figure(figsize=(15, 6))
    
    x = np.arange(len(tokens))
    bars = plt.bar(x, saliency, color='skyblue')
    
    if highlight_ranges:
        for start, end in highlight_ranges:
            # Shift by 1 for [CLS] token and adjust for 1-based indexing if needed
            # Assuming start/end are 1-based indices for the protein sequence
            # tokens[0] is [CLS], tokens[1] is seq[0]
            for i in range(start, end + 1):
                if i < len(bars):
                    bars[i].set_color('orange')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Residue Saliency'),
                          Patch(facecolor='orange', label='Known Structural Epitope')]
        plt.legend(handles=legend_elements)

    plt.xticks(x, tokens, rotation=90, fontsize=8)
    plt.title(title)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Interpret ESM Allergenicity Model")
    parser.add_argument("--model_path", type=str, default="best_model.pt", help="Path to fine-tuned model")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="Backbone model name")
    parser.add_argument("--sequence", type=str, required=True, help="Protein sequence to analyze")
    parser.add_argument("--output_dir", type=str, default="interpret_results", help="Directory to save plots")
    parser.add_argument("--epitopes", type=str, default=None, help="Comma-separated ranges to highlight (e.g., 46-52,146-159)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    highlight_ranges = []
    if args.epitopes:
        for r in args.epitopes.split(','):
            start, end = map(int, r.split('-'))
            highlight_ranges.append((start, end))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_model(args.model_path, args.model_name, device)
    
    print(f"Analyzing sequence (length {len(args.sequence)})...")
    attentions, saliency, tokens, score = get_attentions_and_saliency(model, tokenizer, args.sequence, device)
    
    print(f"Predicted Allergenicity Score: {score:.4f}")
    
    # Save saliency plot
    plot_saliency(saliency, tokens, os.path.join(args.output_dir, "saliency_map.png"), 
                  title=f"Residue Importance (Score: {score:.4f})",
                  highlight_ranges=highlight_ranges)
    
    # Visualize top attention heads in final layers (30-33 for 650M)
    # Note: layers are 0-indexed.
    num_layers = len(attentions)
    target_layers = range(max(0, num_layers-4), num_layers)
    
    for layer_idx in target_layers:
        # Just visualize head 0 as an example, or could search for heads with long-range contacts
        plot_attention_heatmap(attentions[layer_idx][0], tokens, layer_idx, 0, 
                             os.path.join(args.output_dir, f"attention_L{layer_idx}_H0.png"))

    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
