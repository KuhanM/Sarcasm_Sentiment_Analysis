import pandas as pd
from sklearn.model_selection import train_test_split

print("="*70)
print("STEP 4: CREATING TRAIN/VAL/TEST SPLITS")
print("="*70)

# Load dataset
df = pd.read_csv('combined_dataset_with_sentiment.csv')
print(f"\nTotal samples: {len(df):,}")

# Stratified split (maintain class balance)
# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df, 
    test_size=0.3, 
    random_state=42,
    stratify=df['sarcasm']
)

# Second split: 15% val, 15% test (from 30% temp)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['sarcasm']
)

print(f"\n✓ Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"✓ Val set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"✓ Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

# Verify balance
print("\n" + "="*70)
print("CLASS BALANCE VERIFICATION")
print("="*70)

for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    print(f"\n{split_name} Set:")
    print(f"  Sarcasm: {split_df['sarcasm'].mean()*100:.1f}%")
    print(f"  Sentiment distribution:")
    for sent in [0, 1, 2]:
        pct = (split_df['sentiment'] == sent).mean() * 100
        print(f"    {['Negative', 'Neutral', 'Positive'][sent]}: {pct:.1f}%")

# Save splits
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("\n" + "="*70)
print("✅ SPLITS SAVED!")
print("="*70)
print("Files created:")
print("  - data/train.csv")
print("  - data/val.csv")
print("  - data/test.csv")
print("="*70)
