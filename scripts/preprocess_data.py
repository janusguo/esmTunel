import os
import subprocess
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import argparse

def run_cdhit(input_fasta, output_fasta, identity=0.9):
    cmd = f"cd-hit -i {input_fasta} -o {output_fasta} -c {identity} -n 5 -M 16000 -d 0"
    print(f"Running CD-HIT: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def process_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    allergen_file = os.path.join(data_dir, "allergens.fasta")
    non_allergen_file = os.path.join(data_dir, "non_allergens.fasta")
    combined_file = os.path.join(data_dir, "combined.fasta")
    
    # Label and combine
    print("Combining sequences...")
    sequences = []
    with open(combined_file, "w") as f_out:
        if os.path.exists(allergen_file):
            for record in SeqIO.parse(allergen_file, "fasta"):
                record.id = f"allergen_{record.id}"
                SeqIO.write(record, f_out, "fasta")
        else:
            print(f"Warning: {allergen_file} not found")
            
        if os.path.exists(non_allergen_file):
            for record in SeqIO.parse(non_allergen_file, "fasta"):
                record.id = f"nonallergen_{record.id}"
                SeqIO.write(record, f_out, "fasta")
        else:
            print(f"Warning: {non_allergen_file} not found")
            
    # Run CD-HIT
    reduced_file = os.path.join(data_dir, "reduced.fasta")
    run_cdhit(combined_file, reduced_file)
    
    # Load reduced sequences and create dataframe
    print("Processing reduced sequences...")
    data = []
    for record in SeqIO.parse(reduced_file, "fasta"):
        label = 1 if record.id.startswith("allergen_") else 0
        data.append({
            "id": record.id,
            "sequence": str(record.seq),
            "label": label
        })
        
    df = pd.DataFrame(data)
    print(f"Total sequences after redundancy reduction: {len(df)}")
    print(df['label'].value_counts())
    
    # Split data
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, stratify=train_df['label']) # 0.125 * 0.8 = 0.1
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="/home/team/shared/data")
    args = parser.parse_args()
    
    process_data(args.data_dir, args.output_dir)
