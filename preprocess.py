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

    min_sentences = float('inf')
    max_sentences = 0
    min_line_number = None
    max_line_number = None
    line_number = 0
    shortest_sentence_length = float('inf')
    shortest_sentence = ""

    all_sentences = []
    with open(file_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in tqdm(file, desc="Processing lines"):
            line_number += 1
            clean_line = clean_text(line)
            try:
                data = json.loads(clean_line)
                output_file.write(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line: {line}")
                continue  # Skip lines that still have issues after cleaning
            sentences = data['syms']
            num_sentences = len(sentences)
            all_sentences.extend(sentences)

            # Update min and max sentences info
            if num_sentences < min_sentences:
                min_sentences = num_sentences
                min_line_number = line_number
            if num_sentences > max_sentences:
                max_sentences = num_sentences
                max_line_number = line_number

            # Find shortest sentence
            for sentence in sentences:
                if len(sentence) < shortest_sentence_length:
                    shortest_sentence_length = len(sentence)
                    shortest_sentence = sentence

    sampled_sentences = random.sample(all_sentences, 100000) if len(all_sentences) >= 100000 else all_sentences

    with open(sample_output_path, 'w') as sample_file:
        for sentence in sampled_sentences:
            sample_file.write(json.dumps({"document": sentence}) + '\n')

    print(f"Processed {line_number} entries.")
    print(f"Total filtered sentences: {len(all_sentences)}")
    print(f"Line {min_line_number} has the least number of sentences (post-filtering): {min_sentences}")
    print(f"Line {max_line_number} has the most number of sentences (post-filtering): {max_sentences}")
    print(f"Shortest sentence is {shortest_sentence_length} characters long: \"{shortest_sentence}\"")
    return output_path

new_file_path = process_jsonl('/home/sg2323/project/authorship/data/amazon/train.jsonl')
print(f"Filtered data written to {new_file_path}")
