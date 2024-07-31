import torch
import pandas as pd
import random
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from bm25_pt.bm25 import TokenizedBM25
from cluster import cluster_dataset, pad_to_length
import json
import argparse

def load_tensors(path: str) -> dict:
    return torch.load(path)

def batch_tokenize_texts(tokenizer, texts, batch_size=32, max_length=128):
    all_input_ids = []
    all_attention_masks = []
    token_lengths = []
    text_index_by_length = {}

    num_texts = len(texts)
    for idx_start in range(0, num_texts, batch_size):
        idx_end = min(idx_start + batch_size, num_texts)
        batch_texts = texts[idx_start:idx_end]

        tokenized_batch = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized_batch["input_ids"].tolist()
        attention_masks = tokenized_batch["attention_mask"].tolist()

        # Record the length of tokens for each document excluding padding tokens
        for idx, ids in enumerate(input_ids):
            non_padding_length = len([token for token in ids if token != tokenizer.pad_token_id])
            token_lengths.append(non_padding_length)
            # Record the text index corresponding to its token length
            text_index_by_length[non_padding_length] = idx_start + idx

        all_input_ids.extend(input_ids)
        all_attention_masks.extend(attention_masks)

    return all_input_ids, all_attention_masks


def process_clusters(cluster_assignments, documents):
    simplified_assignments = {doc_id: clusters[0] for doc_id, clusters in cluster_assignments.items()}
    cluster_indices = {}
    for doc_id, cluster_id in simplified_assignments.items():
        if cluster_id not in cluster_indices:
            cluster_indices[cluster_id] = []
        cluster_indices[cluster_id].append(doc_id)

    cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in cluster_indices.items()}
    min_docs = min(cluster_sizes.values())
    max_docs = max(cluster_sizes.values())

    print(f"Cluster with the least documents contains: {min_docs} documents")
    print(f"Cluster with the most documents contains: {max_docs} documents")

    for cluster_id in range(6):
        if cluster_id in cluster_indices:
            print(f"Documents from Cluster {cluster_id}:")
            sample_size = min(5, len(cluster_indices[cluster_id]))
            if sample_size > 0:
                sample_indices = random.sample(cluster_indices[cluster_id], sample_size)
                for index in sample_indices:
                    print(documents[index])

def process_action(input_file, tokenizer, bm25_state_path, centroids, start_idx, end_idx, batch_size):
    # Load the BM25 state and prepare BM25 model
    state = load_tensors(bm25_state_path)
    document_input_ids = state['document_input_ids']
    vocab_size = state['vocab_size']
    bm25 = TokenizedBM25(vocab_size=vocab_size)
    bm25.index(document_input_ids)

    # Read the entire file into memory
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Process the specified range of lines
    results = []
    for idx in tqdm(range(start_idx, min(end_idx, len(lines))), desc="Processing documents"):
        data = json.loads(lines[idx])
        test_documents = data["syms"]
        test_input_ids, test_attention_masks = batch_tokenize_texts(tokenizer, test_documents, batch_size=batch_size)

        test_input_ids = [pad_to_length(torch.tensor(t)) for t in test_input_ids]
        test_input_ids = torch.stack(test_input_ids)

        test_bag = bm25.docs_to_bags(test_input_ids).to_sparse_coo().float()
        result = torch.sparse.mm(test_bag, centroids.T).to_dense()
        max_values, max_indices = torch.max(result, dim=1)

        action_types = [str(index.item()) for index in max_indices]
        data["action_type"] = action_types
        lines[idx] = json.dumps(data) + '\n'  # Update the line with new data

    # Write the updated contents back to the file
    with open(input_file, 'w') as file:
        file.writelines(lines)

    print("Processing completed and updated in", input_file)

# def process_actoin(input_file, tokenizer, bm25_state_path, centroids):
#     output_file = input_file.replace(".jsonl", "_out.jsonl")
#     state = load_tensors(bm25_state_path)
#     document_input_ids = state['document_input_ids']
#     vocab_size = state['vocab_size']
#     bm25 = TokenizedBM25(vocab_size=vocab_size)
#     bm25.index(document_input_ids)

#     results = []
#     with open(input_file, 'r') as file:
#         for line in tqdm(file, desc="Processing documents"):
#             data = json.loads(line)
#             test_documents = data["syms"]
#             test_input_ids, test_attention_masks = batch_tokenize_texts(tokenizer, test_documents, batch_size=512)

#             test_input_ids = [pad_to_length(torch.tensor(t)) for t in test_input_ids]
#             test_input_ids = torch.stack(test_input_ids)

#             test_bag = bm25.docs_to_bags(test_input_ids).to_sparse_coo().float()

#             result = torch.sparse.mm(test_bag, centroids.T).to_dense()
#             max_values, max_indices = torch.max(result, dim=1)

#             action_types = [str(index.item()) for index in max_indices]
#             data["action_type"] = action_types
#             results.append(json.dumps(data))

#     # 写入输出文件
#     with open(output_file, 'w') as f_out:
#         f_out.write("\n".join(results))

#     print("Processing completed and saved to", output_file)
    

def main(args):
    # Load dataset
    dataset = load_dataset('json', data_files=args.samples_filename, split='train')
    
    # Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.model_max_length)

    # # Tokenize documents in the dataset
    # input_ids, attention_masks = batch_tokenize_texts(tokenizer, dataset['document'], batch_size=args.batch_size, max_length=args.model_max_length)
    # dataset = dataset.add_column('input_ids', input_ids)
    # dataset = dataset.add_column('attention_mask', attention_masks)

    # # Clustering
    # cluster_assignments, centroids = cluster_dataset(
    #     dataset=dataset,
    #     model="bm25",
    #     query_key='input_ids',
    #     document_key='input_ids',
    #     query_to_doc=True,
    #     cluster_size=args.cluster_size,
    # )
    # with open('centroids.pkl', 'wb') as f:
    #     pickle.dump(centroids, f)

    with open('centroids.pkl', 'rb') as f:
        centroids = pickle.load(f)

    # Process cluster results
    # process_clusters(cluster_assignments, dataset['document'])

    # Load BM25 state
    start_idx = 0
    end_idx = 10000
    process_action(args.input_filename, tokenizer, args.bm25_state_path, centroids, start_idx, end_idx, args.batch_size)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and cluster textual data.")
    parser.add_argument('--samples_filename', type=str, default='/home/sg2323/project/authorship/data/amazon/train_100k_samples.jsonl', help='Path to the samples dataset file.')
    parser.add_argument('--input_filename', type=str, default='/home/sg2323/project/authorship/data/amazon/train_processed.jsonl', help='Path to the input dataset file.')
    parser.add_argument('--model_path', type=str, default='sentence-transformers/paraphrase-distilroberta-base-v1', help='Path to the model used for tokenization.')
    parser.add_argument('--model_max_length', type=int, default=128, help='Maximum length of the tokenized sequences.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size used for tokenization.')
    parser.add_argument('--cluster_size', type=int, default=1000, help='Size of each cluster.')
    parser.add_argument('--bm25_state_path', type=str, default='bm25_state.pth', help='Path to the saved BM25 state file.')
    args = parser.parse_args()
    
    main(args)

