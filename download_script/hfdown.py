from datasets import load_dataset
import pandas as pd
import os

os.makedirs('data/huggingface', exist_ok=True)

print("="*60)
print("DOWNLOADING SARCASM DATASETS FROM HUGGING FACE")
print("="*60)

### DATASET 1: Skip SarcasmNet (deprecated) - Use alternative
print("\n[1/4] Downloading detection-datasets/sarcasm...")
try:
    dataset1 = load_dataset("detection-datasets/sarcasm")
    df1 = pd.DataFrame(dataset1['train'])
    print(f"✓ Detection-datasets: {len(df1):,} samples")
    print(f"  Columns: {df1.columns.tolist()}")
    df1.to_csv('detection_sarcasm.csv', index=False)
    print(f"  Saved to: data/huggingface/detection_sarcasm.csv")
except Exception as e:
    print(f"✗ Error: {e}")

### DATASET 2: nikesh66/Sarcasm-dataset (ALREADY WORKING)
print("\n[2/4] Downloading nikesh66/Sarcasm-dataset...")
try:
    dataset2 = load_dataset("nikesh66/Sarcasm-dataset")
    df2 = pd.DataFrame(dataset2['train'])
    print(f"✓ Nikesh66: {len(df2):,} samples")
    print(f"  Columns: {df2.columns.tolist()}")
    df2.to_csv('nikesh_sarcasm.csv', index=False)
    print(f"  Saved to: data/huggingface/nikesh_sarcasm.csv")
except Exception as e:
    print(f"✗ Error: {e}")

### DATASET 3: Automatic Sarcasm Detection Twitter (ALREADY WORKING)
print("\n[3/4] Downloading shiv213/Automatic-Sarcasm-Detection-Twitter...")
try:
    dataset3 = load_dataset("shiv213/Automatic-Sarcasm-Detection-Twitter")
    df3_train = pd.DataFrame(dataset3['train'])
    df3_test = pd.DataFrame(dataset3['test'])
    print(f"✓ Twitter Sarcasm: {len(df3_train):,} train + {len(df3_test):,} test samples")
    print(f"  Columns: {df3_train.columns.tolist()}")
    df3_train.to_csv('twitter_sarcasm_train.csv', index=False)
    df3_test.to_csv('twitter_sarcasm_test.csv', index=False)
    print(f"  Saved to: data/huggingface/twitter_sarcasm_*.csv")
except Exception as e:
    print(f"✗ Error: {e}")

### DATASET 4: raquiba/Sarcasm (Alternative)
print("\n[4/4] Downloading raquiba/Sarcasm...")
try:
    dataset4 = load_dataset("raquiba/Sarcasm")
    df4 = pd.DataFrame(dataset4['train'])
    print(f"✓ Raquiba: {len(df4):,} samples")
    print(f"  Columns: {df4.columns.tolist()}")
    df4.to_csv('raquiba_sarcasm.csv', index=False)
    print(f"  Saved to: data/huggingface/raquiba_sarcasm.csv")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
