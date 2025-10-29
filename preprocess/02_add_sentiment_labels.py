import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch

print("="*70)
print("STEP 2: ADDING SENTIMENT LABELS (FIXED)")
print("="*70)

# Load cleaned dataset
df = pd.read_csv('combined_dataset_cleaned.csv')
print(f"\nLoaded: {len(df):,} samples")

print("\nInitializing sentiment model...")
device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

# Load model and tokenizer explicitly with safetensors
try:
    print("Loading: cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    # Load model with safetensors (from cached file)
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        use_safetensors=True
    )
    
    # Move to device
    if device >= 0:
        model = model.to('cuda')
    
    # Create pipeline WITHOUT use_safetensors parameter
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    print(f"✓ Model loaded successfully")
    
except Exception as e:
    print(f"Error loading Twitter-RoBERTa: {e}")
    print("\nFalling back to DistilBERT...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

# Function to get sentiment with proper error handling
def get_sentiment_batch(texts, batch_size=32):
    """Process texts in batches and return sentiment labels"""
    sentiments = []
    errors = 0
    
    print(f"\nProcessing {len(texts):,} texts in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
        batch = texts[i:i+batch_size]
        
        try:
            # Call pipeline WITHOUT extra parameters
            results = sentiment_analyzer(batch, truncation=True, max_length=512)
            
            for result in results:
                label = result['label'].upper()
                
                # Map different label formats
                # Twitter-RoBERTa uses: negative, neutral, positive
                # DistilBERT uses: NEGATIVE, POSITIVE
                
                if any(neg in label for neg in ['NEGATIVE', 'NEG', 'LABEL_0']):
                    sentiments.append(0)
                elif any(pos in label for pos in ['POSITIVE', 'POS', 'LABEL_2']):
                    sentiments.append(2)
                elif 'NEUTRAL' in label or 'LABEL_1' in label:
                    sentiments.append(1)
                else:
                    # If unclear, use simple heuristic
                    if result.get('score', 0) > 0.6:
                        sentiments.append(2)  # High confidence = positive
                    elif result.get('score', 0) < 0.4:
                        sentiments.append(0)  # Low confidence = negative
                    else:
                        sentiments.append(1)  # Medium = neutral
                    
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only print first 3 errors
                print(f"\nError in batch {i//batch_size}: {str(e)[:100]}")
            
            # For error batches, process individually
            for text in batch:
                try:
                    result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
                    label = result['label'].upper()
                    
                    if any(neg in label for neg in ['NEGATIVE', 'NEG']):
                        sentiments.append(0)
                    elif any(pos in label for pos in ['POSITIVE', 'POS']):
                        sentiments.append(2)
                    else:
                        sentiments.append(1)
                except:
                    sentiments.append(1)  # Ultimate fallback
    
    if errors > 3:
        print(f"\n⚠ Total errors: {errors} batches")
    
    return sentiments

# Process all texts
batch_size = 32 if device >= 0 else 16
texts = df['text'].tolist()
sentiments = get_sentiment_batch(texts, batch_size=batch_size)

df['sentiment'] = sentiments

# Verify we got real predictions
unique_sentiments = df['sentiment'].nunique()
if unique_sentiments == 1:
    print("\n⚠ WARNING: All predictions are the same class!")
    print("This indicates a problem. Trying single-text processing...")
    
    # Fallback: process first 100 individually to diagnose
    test_samples = texts[:100]
    test_results = []
    for text in tqdm(test_samples, desc="Testing"):
        try:
            result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
            print(f"Text: {text[:50]}... → {result}")
            test_results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            break

# Statistics
print("\n" + "="*70)
print("SENTIMENT LABEL STATISTICS")
print("="*70)

sentiment_dist = df['sentiment'].value_counts().sort_index()
print(f"\nSentiment distribution:")
for idx, label in enumerate(['Negative', 'Neutral', 'Positive']):
    count = sentiment_dist.get(idx, 0)
    pct = count / len(df) * 100
    print(f"  {label} ({idx}): {count:,} ({pct:.1f}%)")

# Check if distribution is reasonable
if sentiment_dist.get(1, 0) > len(df) * 0.95:
    print("\n❌ ERROR: 95%+ samples are Neutral - sentiment labeling FAILED")
    print("Please check model output and try alternative model")
else:
    print("\n✓ Sentiment distribution looks reasonable")

# Cross-tabulation
print(f"\nSarcasm vs Sentiment cross-tabulation:")
crosstab = pd.crosstab(df['sarcasm'], df['sentiment'], 
                       rownames=['Sarcasm'], colnames=['Sentiment'],
                       margins=True)
print(crosstab)

# Sample predictions
print(f"\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

for sarc_val in [0, 1]:
    sarc_label = "Non-Sarcastic" if sarc_val == 0 else "Sarcastic"
    print(f"\n{sarc_label} Examples:")
    
    # Show one sample from each sentiment class
    for sent_val in [0, 1, 2]:
        samples = df[(df['sarcasm'] == sarc_val) & (df['sentiment'] == sent_val)]
        if len(samples) > 0:
            sample = samples.iloc[0]
            sent_label = ['Negative', 'Neutral', 'Positive'][sent_val]
            print(f"  [{sent_label}] {sample['text'][:80]}...")

# Save with sentiment labels
output_path = 'combined_dataset_with_sentiment.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Saved to: {output_path}")
print("="*70)
