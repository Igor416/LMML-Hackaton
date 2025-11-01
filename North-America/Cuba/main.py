"""
Compact Text Processing - Remove Stopwords using NLTK
Preserves original word case and punctuation.
"""

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK data is downloaded
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}") if resource == "punkt" else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))

def remove_stopwords(line):
    """Remove only stopwords from a line, preserving original case and punctuation."""
    tokens = word_tokenize(line)
    filtered = [t for t in tokens if t.lower() not in stop_words]
    
    # Join tokens, but don't add space before punctuation
    result = []
    for i, token in enumerate(filtered):
        if i > 0 and token in string.punctuation:
            # Attach punctuation directly to previous token (no space)
            result[-1] += token
        else:
            result.append(token)
    
    return ' '.join(result)

# Read input, process lines, and write output
with open('input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

filtered_lines = [remove_stopwords(line.rstrip('\n')) for line in lines]

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_lines))

print(f"Processed {len(lines)} lines, output saved to output.txt")
