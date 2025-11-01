"""
Text Processing Script - Remove Stopwords using NLTK

This script reads text from input.txt, removes stopwords using NLTK,
and writes the processed text to output.txt.
Compact Text Processing - Remove Stopwords using NLTK
Preserves original word case.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re


# Download required NLTK data if not already downloaded
def download_nltk_data():
    """Download required NLTK datasets."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
# Ensure required NLTK data is downloaded
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find(f"tokenizers/{resource}") if resource == "punkt" else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print("Downloading NLTK stopwords corpus...")
        nltk.download('stopwords', quiet=True)
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def remove_stopwords_from_line(line):
    """
    Remove stopwords from a single line using NLTK.
    
    Args:
        line (str): Input line to process
        
    Returns:
        str: Line with stopwords removed
    """
    # Skip empty lines
    if not line.strip():
        return ''
    
    # Tokenize the line into words (preserves punctuation as separate tokens)
    tokens = word_tokenize(line)
    
    # Get English stopwords (convert to lowercase for comparison)
    stop_words = set(stopwords.words('english'))
    
    # Filter out only stopwords (preserve punctuation and case)
    filtered_tokens = [
        token for token in tokens 
        if token.lower() not in stop_words
    ]
    
    # Reconstruct the line with spaces between tokens
    # NLTK tokenize separates punctuation, so we need to handle spacing
    filtered_line = ' '.join(filtered_tokens)
    
    # Clean up spacing around punctuation (restore original punctuation spacing)
    filtered_line = re.sub(r'\s+([.,!?;:])', r'\1', filtered_line)  # Remove space before punctuation
    filtered_line = re.sub(r'([.,!?;:])\s+', r'\1 ', filtered_line)  # Ensure space after punctuation
    
    # Capitalize first letter of the line (find first letter and capitalize it)
    if filtered_line:
        for i, char in enumerate(filtered_line):
            if char.isalpha():
                filtered_line = filtered_line[:i] + char.upper() + filtered_line[i+1:]
                break
    
    return filtered_line

def remove_stopwords(line):
    """Remove stopwords from a line, preserving original case and punctuation."""
    tokens = word_tokenize(line)
    filtered = [t for t in tokens if t.lower() not in stop_words]
    result = ' '.join(filtered)
    # Clean up spacing around punctuation
    result = re.sub(r'\s+([.,!?;:])', r'\1', result)
    result = re.sub(r'([.,!?;:])\s+', r'\1 ', result)
    # Capitalize first letter of the line (find first letter and capitalize it)
    if result:
        for i, char in enumerate(result):
            if char.isalpha():
                result = result[:i] + char.upper() + result[i+1:]
                break
    return result

# Read input, process lines, and write output
with open('input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def main():
    """
    Main function to read input.txt, remove stopwords, and write to output.txt.
    """
    # Download required NLTK data
    print("Setting up NLTK...")
    download_nltk_data()
    print("NLTK setup complete.\n")
    
    # Read input file line by line
    print("Reading input.txt...")
    try:
        with open('input.txt', 'r') as f:
            lines = f.readlines()
        print(f"Successfully read {len(lines)} lines from input.txt\n")
    except FileNotFoundError:
        print("Error: input.txt not found!")
        print("Please ensure input.txt exists in the current directory.")
        return
    except Exception as e:
        print(f"Error reading input.txt: {e}")
        return
    
    # Display original text (first 10 lines as preview)
    print("=" * 80)
    print("ORIGINAL TEXT (First 10 lines)")
    print("=" * 80)
    preview_lines = min(10, len(lines))
    for i in range(preview_lines):
        print(lines[i].rstrip())
    if len(lines) > 10:
        print(f"... and {len(lines) - 10} more lines")
    print()
    
    # Process each line to remove stopwords
    print("Processing text to remove stopwords line by line...")
    filtered_lines = []
    for line in lines:
        # Remove stopwords from the line
        filtered_line = remove_stopwords_from_line(line.rstrip('\n'))
        filtered_lines.append(filtered_line)
    
    # Join lines with newlines (each filtered line is without trailing newline)
    filtered_text = '\n'.join(filtered_lines)
    
    # Display filtered text (first 10 lines as preview)
    print("=" * 80)
    print("FILTERED TEXT (First 10 lines - Stopwords Removed)")
    print("=" * 80)
    for i in range(preview_lines):
        print(filtered_lines[i])
    if len(filtered_lines) > 10:
        print(f"... and {len(filtered_lines) - 10} more lines")
    print()
    
    # Write to output file
    print("Writing to output.txt...")
    try:
        with open('output.txt', 'w') as f:
            f.write(filtered_text)
        print(f"Successfully wrote {len(filtered_text)} characters to output.txt")
    except Exception as e:
        print(f"Error writing to output.txt: {e}")
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    # Count words across all lines
    original_words = sum(len(word_tokenize(line)) for line in lines)
    filtered_words = sum(len(word_tokenize(line)) for line in filtered_lines)
    removed_words = original_words - filtered_words
    
    print(f"Original line count: {len(lines)}")
    print(f"Filtered line count: {len(filtered_lines)}")
    print(f"Original word count: {original_words}")
    print(f"Filtered word count: {filtered_words}")
    print(f"Stopwords removed: {removed_words}")
    print("=" * 80)
filtered_lines = [remove_stopwords(line.rstrip('\n')) for line in lines]

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(filtered_lines))

if __name__ == "__main__":
    main()
print(f"Processed {len(lines)} lines, output saved to output.txt")
