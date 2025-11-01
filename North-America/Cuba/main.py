"""
Compact Text Processing - Remove Stopwords using NLTK
Preserves original word case.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure required NLTK data is downloaded
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}") if resource == "punkt" else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def remove_stopwords(line):
    """Remove stopwords and punctuation from a line, preserving original case."""
    tokens = word_tokenize(line)
    filtered = [t for t in tokens if t.lower() not in stop_words and t not in punctuation]
    return ' '.join(filtered)

# Read input, process lines, and write output
with open('input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

filtered_lines = [remove_stopwords(line.rstrip('\n')) for line in lines]

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_lines))

print(f"Processed {len(lines)} lines, output saved to output.txt")
