from datasets import load_dataset
import pandas as pd
import os

# Create directory for datasets
os.makedirs('data/huggingface', exist_ok=True)

print("="*60)
print("DOWNLOADING SARCASM DATASETS FROM HUGGING FACE")
print("="*60)

### DATASET 1: SarcasmNet/sarcasm
print("\n[1/3] Downloading SarcasmNet/sarcasm...")
try:
    dataset = load_dataset("SarcasmNet/sarcasm", trust_remote_code=True)
    import pandas as pd
    df = pd.DataFrame(dataset['train'])
    df.to_csv('data/huggingface/sarcasmnet.csv', index=False)
    print(f"✓ SarcasmNet: {len(df):,} samples")
except Exception as e:
    print(f"Still failing. Using alternative dataset instead.")

# ### DATASET 2: nikesh66/Sarcasm-dataset
# print("\n[2/3] Downloading nikesh66/Sarcasm-dataset...")
# try:
#     dataset2 = load_dataset("nikesh66/Sarcasm-dataset")
#     df2 = pd.DataFrame(dataset2['train'])
#     print(f"✓ Nikesh66: {len(df2):,} samples")
#     print(f"  Columns: {df2.columns.tolist()}")
    
#     # Save to CSV
#     df2.to_csv('nikesh_sarcasm.csv', index=False)
#     print(f"  Saved to: data/huggingface/nikesh_sarcasm.csv")
# except Exception as e:
#     print(f"✗ Error: {e}")

# ### DATASET 3: Automatic Sarcasm Detection Twitter
# print("\n[3/3] Downloading shiv213/Automatic-Sarcasm-Detection-Twitter...")
# try:
#     dataset3 = load_dataset("shiv213/Automatic-Sarcasm-Detection-Twitter")
#     df3 = pd.DataFrame(dataset3['train'])
#     print(f"✓ Twitter Sarcasm: {len(df3):,} samples")
#     print(f"  Columns: {df3.columns.tolist()}")
    
#     # Save to CSV
#     df3.to_csv('twitter_sarcasm_auto.csv', index=False)
#     print(f"  Saved to: data/huggingface/twitter_sarcasm_auto.csv")
# except Exception as e:
#     print(f"✗ Error: {e}")

# print("\n" + "="*60)
# print("DOWNLOAD COMPLETE!")
# print("="*60)
