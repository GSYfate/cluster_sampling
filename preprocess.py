import json
import os
import random
from tqdm import tqdm
import re

def clean_text(text):
    # Remove null character and other potential control characters if needed
    return text.replace('\x00', '').strip()

def is_valid_sentence(sentence):
    # Check if the sentence contains any unicode characters
    if re.search(r'[^\x00-\x7F]', sentence):
        return False
    # Check if the sentence contains repeating word patterns
    if re.match(r'^(?:\b(\w+)\s+\1\b)+$', sentence):
        return False
    return True

def process_jsonl(file_path):
    output_path = '/home/sg2323/project/authorship/cluster_sampling/data/reddit/train_processed_test.jsonl'
    sample_output_path =  '/home/sg2323/project/authorship/cluster_sampling/data/reddit/train_1000_samples_200.jsonl'
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    line_number = 0
    all_sentences = []
    min_sentences = float('inf')
    max_sentences = 0
    min_length = float('inf')
    max_length = 0

    with open(file_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in tqdm(file, desc="Processing lines"):
            line_number += 1
            if line_number >2000:
                break
            clean_line = clean_text(line)
            try:
                data = json.loads(clean_line)
                # Filter sentences and collect valid indices
                valid_indices = [i for i, s in enumerate(data['syms']) 
                                 if len(s) >= 200 and is_valid_sentence(s) and len(s.split()) > 20]
                filtered_sentences = [data['syms'][i] for i in valid_indices]

                # Check the length of the filtered sentences list
                num_sentences = len(filtered_sentences)
                if num_sentences < 4:
                    continue  # Skip writing to the output if less than 4 valid sentences

                data['syms'] = filtered_sentences
                output_file.write(json.dumps(data) + '\n')
                all_sentences.extend(filtered_sentences)

                # Update minimum and maximum counts
                min_sentences = min(min_sentences, num_sentences)
                max_sentences = max(max_sentences, num_sentences)

                for sentence in filtered_sentences:
                    words = len(sentence.split())
                    min_length = min(min_length, words)
                    max_length = max(max_length, words)

            except json.JSONDecodeError:
                print(f"Error decoding JSON on line: {line}")
                continue  # Skip lines that still have issues after cleaning

    sampled_sentences = random.sample(all_sentences, 1000) if len(all_sentences) >= 1000 else all_sentences
    with open(sample_output_path, 'w') as sample_file:
        for sentence in sampled_sentences:
            sample_file.write(json.dumps({"document": sentence}) + '\n')

    print(f"Processed {line_number} entries.")
    print(f"Total filtered sentences: {len(all_sentences)}")
    print(f"Minimum sentences count in any valid entry: {min_sentences if min_sentences != float('inf') else 'No valid entries'}")
    print(f"Maximum sentences count in any valid entry: {max_sentences}")
    print(f"Minimum number of words in sentences after final filtering: {min_length if min_length != float('inf') else 'No valid sentences'}")
    print(f"Maximum number of words in sentences after final filtering: {max_length}")
    return output_path

# Example usage

new_file_path = process_jsonl('/share/rush/authorship/data/all_pii/reddit2020.072524.jsonl')
print(f"Filtered data written to {new_file_path}")
