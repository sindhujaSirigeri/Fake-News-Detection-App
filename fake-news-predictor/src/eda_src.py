# Import Libraries
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import os

# Load Serialised Data

# Load the processed DataFrame
df = pd.read_hdf('../data/processed/processed_data.h5', key='df')

# Load the padded sequences
with h5py.File('../data/processed/padded_sequences.h5', 'r') as f:
    padded_sequences = f['padded_sequences'][:]

# Load the tokenizer
with open('../data/processed/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Display basic information about the DataFrame
print(df.info())

# Exploratory Data Analysis Steps

# 1. Class Distribution Analysis
sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.show()

# Display the actual counts
print(df['label'].value_counts())

# 2. Text Length Distribution
df['text_length'] = df['cleaned_text'].apply(len)

plt.hist(df['text_length'], bins=50)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Display summary statistics for text lengths
print(df['text_length'].describe())

# 3. Word Cloud
fake_news_text = ' '.join(df[df['label'] == 0]['cleaned_text'])
real_news_text = ' '.join(df[df['label'] == 1]['cleaned_text'])

fake_wordcloud = WordCloud(max_words=100, background_color='white').generate(fake_news_text)
real_wordcloud = WordCloud(max_words=100, background_color='white').generate(real_news_text)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Fake News')

plt.subplot(1, 2, 2)
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Real News')

plt.show()

# 4. Most Common Words
fake_words = ' '.join(df[df['label'] == 0]['cleaned_text']).split()
real_words = ' '.join(df[df['label'] == 1]['cleaned_text']).split()

fake_word_counts = Counter(fake_words).most_common(10)
real_word_counts = Counter(real_words).most_common(10)

print("Most common words in fake news:", fake_word_counts)
print("Most common words in real news:", real_word_counts)

# 5. Check for Duplicate Entries
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Identify and review duplicates
duplicate_rows = df[df.duplicated()]
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
print(duplicate_rows.head())

# Remove exact duplicates
df_cleaned = df.drop_duplicates()

# Confirm removal
print(f"Number of rows after removing duplicates: {df_cleaned.shape[0]}")

# 6. Correlation Analysis
sns.boxplot(x='label', y='text_length', data=df)
plt.title('Text Length vs Label')
plt.show()

# 7. Visualising Data Distribution
sns.histplot(data=df, x='text_length', hue='label', multiple='stack', bins=50)
plt.title('Text Length Distribution by Label')
plt.show()

# Transformation

# Apply a log transformation to reduce skewness
df['log_text_length'] = np.log1p(df['text_length'])

# Plot the distribution after log transformation
plt.hist(df['log_text_length'], bins=50)
plt.title('Distribution of Log Transformed Text Lengths')
plt.xlabel('Log(Text Length)')
plt.ylabel('Frequency')
plt.show()

# Optionally, you can inspect the distribution statistics
print(df['log_text_length'].describe())

# Save the DataFrame as an HDF5 File

# Create the directory if it doesn't exist
os.makedirs('../data/processed', exist_ok=True)

# Save the cleaned DataFrame to an HDF5 file
df.to_hdf('../data/processed/eda_processed_data.h5', key='df', mode='w')

print("EDA processed DataFrame has been saved to 'data/processed/eda_processed_data.h5'.")
