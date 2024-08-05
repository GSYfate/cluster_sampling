import json
import os
import random
from tqdm import tqdm

def clean_text(text):
    # Remove null character and other potential control characters if needed
    return text.replace('\x00', '').strip()

def process_jsonl(file_path):
    output_path = file_path.replace('.jsonl', '_processed.jsonl')
    sample_output_path = file_path.replace('.jsonl', '_100k_samples.jsonl')
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    line_number = 0
    min_sentences = float('inf')  

    all_sentences = []
    with open(file_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in tqdm(file, desc="Processing lines"):
            line_number += 1
            clean_line = clean_text(line)
            try:
                data = json.loads(clean_line)
                # Filter sentences and collect valid indices
                valid_indices = [i for i, s in enumerate(data['syms']) if len(s) >= 50]
                filtered_sentences = [data['syms'][i] for i in valid_indices]
                data['syms'] = filtered_sentences
                data['hour'] = [data['hour'][i] for i in valid_indices]
                data['minute'] = [data['minute'][i] for i in valid_indices]
                data['day'] = [data['day'][i] for i in valid_indices]
                data['action_type'] = [data['action_type'][i] for i in valid_indices]
                num_sentences = len(filtered_sentences)
                if num_sentences < min_sentences:
                    min_sentences = num_sentences

                if data['syms']:
                    output_file.write(json.dumps(data) + '\n')
                    all_sentences.extend(filtered_sentences)

            except json.JSONDecodeError:
                print(f"Error decoding JSON on line: {line}")
                continue  # Skip lines that still have issues after cleaning

    sampled_sentences = random.sample(all_sentences, 100000) if len(all_sentences) >= 100000 else all_sentences
    with open(sample_output_path, 'w') as sample_file:
        for sentence in sampled_sentences:
            sample_file.write(json.dumps({"document": sentence}) + '\n')

    print(f"Processed {line_number} entries.")
    print(f"Total filtered sentences: {len(all_sentences)}")
    print(f"Minimum number of sentences per line after filtering: {min_sentences}")
    return output_path

# Example usage
new_file_path = process_jsonl('/home/sg2323/project/authorship/cluster_sampling/data/reddit/train.jsonl')
print(f"Filtered data written to {new_file_path}")
