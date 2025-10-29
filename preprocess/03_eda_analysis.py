import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*70)

# Load dataset
df = pd.read_csv('combined_dataset_with_sentiment.csv')
print(f"\nTotal samples: {len(df):,}")

# Create output directory
import os
os.makedirs('results/eda', exist_ok=True)

# 1. Text Length Analysis
print("\n[1/6] Text Length Analysis...")
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nText Length Statistics:")
print(df[['text_length', 'word_count']].describe())

# Plot text length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist([df[df['sarcasm']==1]['text_length'], 
              df[df['sarcasm']==0]['text_length']], 
             bins=50, alpha=0.7, label=['Sarcastic', 'Non-sarcastic'])
axes[0].set_xlabel('Text Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Text Length Distribution')
axes[0].legend()

axes[1].hist([df[df['sarcasm']==1]['word_count'], 
              df[df['sarcasm']==0]['word_count']], 
             bins=50, alpha=0.7, label=['Sarcastic', 'Non-sarcastic'])
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Word Count Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/eda/01_text_length_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_text_length_distribution.png")
plt.close()

# 2. Class Balance
print("\n[2/6] Class Balance Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sarcasm balance
sarcasm_counts = df['sarcasm'].value_counts()
axes[0].bar(['Non-sarcastic', 'Sarcastic'], 
            [sarcasm_counts[0], sarcasm_counts[1]], 
            color=['#3498db', '#e74c3c'])
axes[0].set_ylabel('Count')
axes[0].set_title('Sarcasm Class Balance')
axes[0].set_ylim(0, max(sarcasm_counts) * 1.1)
for i, v in enumerate([sarcasm_counts[0], sarcasm_counts[1]]):
    axes[0].text(i, v + 500, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom')

# Sentiment balance
sentiment_counts = df['sentiment'].value_counts().sort_index()
axes[1].bar(['Negative', 'Neutral', 'Positive'], 
            sentiment_counts.values, 
            color=['#e74c3c', '#95a5a6', '#2ecc71'])
axes[1].set_ylabel('Count')
axes[1].set_title('Sentiment Class Balance')
axes[1].set_ylim(0, max(sentiment_counts) * 1.1)
for i, v in enumerate(sentiment_counts.values):
    axes[1].text(i, v + 500, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/eda/02_class_balance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_class_balance.png")
plt.close()

# 3. Sarcasm vs Sentiment Heatmap
print("\n[3/6] Sarcasm vs Sentiment Analysis...")

crosstab = pd.crosstab(df['sarcasm'], df['sentiment'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlOrRd', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Non-sarcastic', 'Sarcastic'],
            cbar_kws={'label': 'Percentage (%)'})
ax.set_xlabel('Sentiment')
ax.set_ylabel('Sarcasm')
ax.set_title('Sarcasm vs Sentiment Distribution (%)')
plt.tight_layout()
plt.savefig('results/eda/03_sarcasm_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_sarcasm_sentiment_heatmap.png")
plt.close()

# 4. Source Distribution
print("\n[4/6] Source Distribution Analysis...")

source_stats = df.groupby('source').agg({
    'text': 'count',
    'sarcasm': 'sum'
}).rename(columns={'text': 'total', 'sarcasm': 'sarcastic'})
source_stats['non_sarcastic'] = source_stats['total'] - source_stats['sarcastic']

fig, ax = plt.subplots(figsize=(10, 6))
source_stats[['sarcastic', 'non_sarcastic']].plot(kind='bar', stacked=True, 
                                                   color=['#e74c3c', '#3498db'],
                                                   ax=ax)
ax.set_xlabel('Source')
ax.set_ylabel('Count')
ax.set_title('Dataset Samples by Source')
ax.legend(['Sarcastic', 'Non-sarcastic'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/eda/04_source_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_source_distribution.png")
plt.close()

# 5. Word Clouds
print("\n[5/6] Generating Word Clouds...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sarcastic text word cloud
sarcastic_text = ' '.join(df[df['sarcasm']==1]['text'])
wordcloud_sarcastic = WordCloud(width=800, height=400, 
                                 background_color='white',
                                 colormap='Reds').generate(sarcastic_text)
axes[0].imshow(wordcloud_sarcastic, interpolation='bilinear')
axes[0].set_title('Sarcastic Text Word Cloud', fontsize=16)
axes[0].axis('off')

# Non-sarcastic text word cloud
non_sarcastic_text = ' '.join(df[df['sarcasm']==0]['text'])
wordcloud_non_sarcastic = WordCloud(width=800, height=400, 
                                      background_color='white',
                                      colormap='Blues').generate(non_sarcastic_text)
axes[1].imshow(wordcloud_non_sarcastic, interpolation='bilinear')
axes[1].set_title('Non-Sarcastic Text Word Cloud', fontsize=16)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/eda/05_wordclouds.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_wordclouds.png")
plt.close()

# 6. Summary Statistics
print("\n[6/6] Generating Summary Report...")

summary = f"""
{'='*70}
DATASET SUMMARY REPORT
{'='*70}

1. DATASET SIZE
   Total samples: {len(df):,}
   
2. SARCASM LABELS
   Sarcastic: {df['sarcasm'].sum():,} ({df['sarcasm'].mean()*100:.1f}%)
   Non-sarcastic: {(len(df) - df['sarcasm'].sum()):,} ({(1-df['sarcasm'].mean())*100:.1f}%)
   
3. SENTIMENT LABELS
   Negative: {(df['sentiment']==0).sum():,} ({(df['sentiment']==0).mean()*100:.1f}%)
   Neutral: {(df['sentiment']==1).sum():,} ({(df['sentiment']==1).mean()*100:.1f}%)
   Positive: {(df['sentiment']==2).sum():,} ({(df['sentiment']==2).mean()*100:.1f}%)
   
4. TEXT STATISTICS
   Avg text length: {df['text_length'].mean():.1f} characters
   Avg word count: {df['word_count'].mean():.1f} words
   Min words: {df['word_count'].min()}
   Max words: {df['word_count'].max()}
   
5. DATA SOURCES
{df['source'].value_counts().to_string()}

6. SARCASM IN SENTIMENT CATEGORIES
{pd.crosstab(df['sentiment'], df['sarcasm'], 
             rownames=['Sentiment'], colnames=['Sarcasm'],
             margins=True).to_string()}

{'='*70}
"""

with open('results/eda/summary_report.txt', 'w') as f:
    f.write(summary)

print(summary)
print("\n✓ Saved: summary_report.txt")
print("\n" + "="*70)
print("✅ EDA COMPLETE! Check results/eda/ folder for visualizations")
print("="*70)
