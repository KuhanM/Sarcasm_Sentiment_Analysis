import pandas as pd
import json
import re
from tqdm import tqdm

print("="*70)
print("STEP 1: COMBINING AND STANDARDIZING DATASETS")
print("="*70)

# Text cleaning function
def clean_text(text):
    """Clean tweet/headline text"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?\'-]', '', text)
    return text.strip()

datasets = []

# 1. News Headlines
print("\n[1/4] Loading News Headlines...")
news_data = []
with open(r'C:\Kuhan\Paper1\sarcasm-mtl-research\Sarcasm_Headlines_Dataset_v2.json', 'r', encoding='utf-8') as f:
    for line in f:
        news_data.append(json.loads(line))

news_df = pd.DataFrame(news_data)
news_df['text'] = news_df['headline'].apply(clean_text)
news_df['sarcasm'] = news_df['is_sarcastic']
news_df['source'] = 'news'
news_df = news_df[['text', 'sarcasm', 'source']]
print(f"✓ Loaded: {len(news_df):,} samples")
datasets.append(news_df)

# 2. nikesh66 Twitter
print("\n[2/4] Loading nikesh66 Twitter...")
nikesh_df = pd.read_csv(r'C:\Kuhan\Paper1\sarcasm-mtl-research\nikesh_sarcasm.csv', encoding='utf-8')
nikesh_df['text'] = nikesh_df['Tweet'].apply(clean_text)
nikesh_df['sarcasm'] = (nikesh_df['Sarcasm (yes/no)'] == 'yes').astype(int)
nikesh_df['source'] = 'twitter_nikesh'
nikesh_df = nikesh_df[['text', 'sarcasm', 'source']]
print(f"✓ Loaded: {len(nikesh_df):,} samples")
datasets.append(nikesh_df)

# 3. shiv213 Twitter
print("\n[3/4] Loading shiv213 Twitter...")
shiv_train = pd.read_csv(r'C:\Kuhan\Paper1\sarcasm-mtl-research\twitter_sarcasm_train.csv', encoding='utf-8')
shiv_test = pd.read_csv(r'C:\Kuhan\Paper1\sarcasm-mtl-research\twitter_sarcasm_test.csv', encoding='utf-8')
shiv_df = pd.concat([shiv_train, shiv_test], ignore_index=True)
shiv_df['text'] = shiv_df['response'].apply(clean_text)
shiv_df['sarcasm'] = (shiv_df['label'] == 'SARCASM').astype(int)
shiv_df['source'] = 'twitter_shiv'
shiv_df = shiv_df[['text', 'sarcasm', 'source']]
print(f"✓ Loaded: {len(shiv_df):,} samples")
datasets.append(shiv_df)

# 4. ATRK/Figshare Twitter
print("\n[4/4] Loading ATRK/Figshare Twitter...")
atrk_df = pd.read_csv(r'C:\Kuhan\Paper1\sarcasm-mtl-research\ATRK dataset.csv', encoding='latin-1')
atrk_df['text'] = atrk_df['content'].apply(clean_text)
# First 15k are sarcastic, last 15k are non-sarcastic (per Figshare docs)
atrk_df['sarcasm'] = [1] * 15000 + [0] * 15000
atrk_df['source'] = 'twitter_figshare'
atrk_df = atrk_df[['text', 'sarcasm', 'source']]
print(f"✓ Loaded: {len(atrk_df):,} samples")
datasets.append(atrk_df)

# Combine all datasets
print("\n" + "="*70)
print("COMBINING DATASETS")
print("="*70)

combined_df = pd.concat(datasets, ignore_index=True)
print(f"\n✓ Total combined: {len(combined_df):,} samples")

# Remove duplicates
print(f"\nChecking for duplicates...")
duplicates = combined_df['text'].duplicated().sum()
print(f"  Found {duplicates:,} duplicates ({duplicates/len(combined_df)*100:.2f}%)")

combined_df = combined_df.drop_duplicates(subset='text', keep='first')
print(f"  After removal: {len(combined_df):,} samples")

# Remove empty texts
empty_texts = (combined_df['text'].str.strip() == '').sum()
if empty_texts > 0:
    print(f"\nRemoving {empty_texts} empty texts...")
    combined_df = combined_df[combined_df['text'].str.strip() != '']

# Remove very short texts (< 3 words)
combined_df['word_count'] = combined_df['text'].str.split().str.len()
short_texts = (combined_df['word_count'] < 3).sum()
if short_texts > 0:
    print(f"Removing {short_texts} very short texts (< 3 words)...")
    combined_df = combined_df[combined_df['word_count'] >= 3]

combined_df = combined_df.drop('word_count', axis=1)

# Final statistics
print("\n" + "="*70)
print("FINAL STATISTICS")
print("="*70)
print(f"\nTotal samples: {len(combined_df):,}")
print(f"Sarcastic: {combined_df['sarcasm'].sum():,} ({combined_df['sarcasm'].mean()*100:.1f}%)")
print(f"Non-sarcastic: {(len(combined_df) - combined_df['sarcasm'].sum()):,}")

print(f"\nBy source:")
print(combined_df.groupby('source').agg({
    'text': 'count',
    'sarcasm': ['sum', 'mean']
}).round(3))

# Save combined dataset
combined_df.to_csv('combined_dataset_cleaned.csv', index=False)
print(f"\n✓ Saved to: data/combined_dataset_cleaned.csv")
print("="*70)
