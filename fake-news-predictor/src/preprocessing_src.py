# Import Libraries

import os
import re
import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download the stopwords dataset
nltk.download('stopwords')

# Download required resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def remove_stopwords(text):
    word_tokens = text.split()
    filtered_text = [
        word for word in word_tokens if word.lower() not in stop_words
    ]
    return ' '.join(filtered_text)


def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text):
    lemmatized_text = ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split()]
    )
    return lemmatized_text


# Load Dataset

# Load dataset
df = pd.read_csv('../data/raw/WELFake_Dataset.csv')

# Drop the 'Unnamed: 0' column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Handling Missing Values
df.dropna(subset=['text'], inplace=True)
df['title'].fillna('No Title', inplace=True)

# Combine 'title' and 'text' into a single column called 'combined_text'
df['combined_text'] = df['title'].fillna('') + " " + df['text']

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Apply the remove_stopwords function to the 'combined_text' column
df['cleaned_text'] = df['combined_text'].apply(remove_stopwords)

# Apply the cleaning function to the 'cleaned_text' column
df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# Apply the lemmatization function to the 'cleaned_text' column
df['cleaned_text'] = df['cleaned_text'].apply(lemmatize_text)

# Convert all text to lowercase
df['cleaned_text'] = df['cleaned_text'].str.lower()

# Tokenization and Padding

# Initialize the tokenizer without limiting the number of words
tokenizer = Tokenizer()

# Fit the tokenizer on the cleaned text data
tokenizer.fit_on_texts(df['cleaned_text'])

# Get the total number of unique words
total_unique_words = len(tokenizer.word_index)
print(f"Total unique words in the dataset: {total_unique_words}")

# Get the word index (word -> integer mapping) and word counts
word_index = tokenizer.word_index
word_counts = tokenizer.word_counts

# Sort words by frequency
sorted_word_counts = sorted(
    word_counts.items(), key=lambda x: x[1], reverse=True
)

# Calculate the cumulative coverage
cumulative_coverage = []
cumulative_count = 0
total_word_count = sum(word_counts.values())

for word, count in sorted_word_counts:
    cumulative_count += count
    cumulative_coverage.append(cumulative_count / total_word_count)

# Calculate num_words for 90% and 95% coverage
coverage_90 = next(
    i for i, coverage in enumerate(cumulative_coverage) if coverage >= 0.90
) + 1
coverage_95 = next(
    i for i, coverage in enumerate(cumulative_coverage) if coverage >= 0.95
) + 1

print(f"Number of words covering 90% of the dataset: {coverage_90}")
print(f"Number of words covering 95% of the dataset: {coverage_95}")

# Plot the cumulative coverage
plt.plot(cumulative_coverage)
plt.xlabel("Number of words")
plt.ylabel("Cumulative coverage")
plt.title("Cumulative Word Coverage")
plt.axvline(
    x=coverage_90, color='r', linestyle='--',
    label=f'90% coverage ({coverage_90} words)'
)
plt.axvline(
    x=coverage_95, color='g', linestyle='--',
    label=f'95% coverage ({coverage_95} words)'
)
plt.legend()
plt.show()

# Tokenization with a limited vocabulary size
tokenizer = Tokenizer(num_words=24016, oov_token='<OOV>')

# Fit the tokenizer on the cleaned text data
tokenizer.fit_on_texts(df['cleaned_text'])

# Convert the cleaned text to sequences
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

# Calculate the lengths of all sequences
sequence_lengths = [len(seq) for seq in sequences]

# Determine the 90th and 95th percentiles
percentile_90 = int(np.percentile(sequence_lengths, 90))
percentile_95 = int(np.percentile(sequence_lengths, 95))

print(f"90th percentile max_sequence_length set to: {percentile_90}")
print(f"95th percentile max_sequence_length set to: {percentile_95}")

# Pad the sequences to the 95th percentile length
padded_sequences = pad_sequences(
    sequences, maxlen=percentile_95, padding='post', truncating='post'
)

# Verify the padding
print(f"First padded sequence: {padded_sequences[0]}")
print(f"Shape of padded_sequences: {padded_sequences.shape}")

# Plot the distribution of sequence lengths
plt.figure(figsize=(10, 6))
plt.hist(
    sequence_lengths,
    bins=50,
    alpha=0.75,
    color='blue',
    edgecolor='black'
)
plt.axvline(
    percentile_90, color='orange', linestyle='dashed', linewidth=2,
    label=f'90th Percentile: {percentile_90}'
)
plt.axvline(
    percentile_95, color='red', linestyle='dashed', linewidth=2,
    label=f'95th Percentile: {percentile_95}'
)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Sequence Lengths with Percentiles')
plt.legend()
plt.show()

# Save processed data

# Create the directory if it does not exist
os.makedirs('../data/processed', exist_ok=True)

# Save the padded sequences in HDF5 format
with h5py.File('../data/processed/padded_sequences.h5', 'w') as f:
    f.create_dataset('padded_sequences', data=padded_sequences)

# Save the tokenizer separately as Pickle
with open('../data/processed/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save the processed DataFrame
df.to_hdf('../data/processed/processed_data.h5', key='df', mode='w')

print("Data has been serialized and saved in the '../data/processed' folder.")
