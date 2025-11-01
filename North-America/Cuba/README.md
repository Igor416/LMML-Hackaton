# Text Processing Script - Remove Stopwords

## Overview
This script processes text files by removing stopwords using NLTK (Natural Language Toolkit). It reads from `input.txt`, processes each line to remove English stopwords and punctuation, and writes the filtered text to `output.txt`.

## Features
- **Stopword Removal**: Removes common English stopwords using NLTK
- **Punctuation Filtering**: Removes punctuation marks from the text
- **Line-by-Line Processing**: Preserves line structure while filtering
- **Progress Display**: Shows original and filtered text previews
- **Statistics**: Provides summary statistics about the filtering process

## Usage
```bash
# Install dependencies
pip install -r r.txt

# Run the script
python main.py
```

## Requirements
- **NLTK**: Natural Language Toolkit for stopword removal and tokenization
- **Python 3.x**: Required Python version

The script will automatically download required NLTK data (punkt tokenizer and stopwords corpus) on first run.

## Input/Output
- **Input**: `input.txt` - Text file with one sentence per line
- **Output**: `output.txt` - Processed text with stopwords and punctuation removed

## How It Works
1. Downloads required NLTK data if not already present
2. Reads `input.txt` line by line
3. Tokenizes each line into words
4. Filters out:
   - English stopwords (the, a, an, and, etc.)
   - Punctuation marks
5. Rejoins filtered tokens back into lines
6. Writes the processed text to `output.txt`
7. Displays summary statistics

## Example
**Input:**
```
The quick brown fox jumps over the lazy dog.
Hello, world! How are you?
```

**Output:**
```
quick brown fox jumps lazy dog
Hello world How
```

## Notes
- Empty lines are preserved as empty lines in the output
- The script processes text line by line to maintain structure
- Only English stopwords are removed (as defined by NLTK)

