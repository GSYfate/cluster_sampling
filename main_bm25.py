import torch
import random
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from bm25_pt.bm25 import TokenizedBM25
from cluster import cluster_dataset, pad_to_length
import json
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
print(f"Using device: {device}") #


def load_tensors(path: str) -> dict:
    return torch.load(path, map_location=device)


def batch_tokenize_texts(tokenizer, texts, batch_size=8, max_length=128):
    tokenizer.pad_token_id = 0
    all_input_ids = []
    all_attention_masks = []
    token_lengths = []
    text_index_by_length = {}

    num_texts = len(texts)
    for idx_start in tqdm(range(0, num_texts, batch_size)):
        print(idx_start)
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
    sorted_cluster_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1])  # Sorting clusters by size
    min_docs = min(cluster_sizes.values())
    max_docs = max(cluster_sizes.values())
    print("Sorted cluster sizes:")
    for cluster_id, size in sorted_cluster_sizes:
        print(f"Cluster {cluster_id}: {size} documents")

    print(f"Cluster with the least documents contains: {min_docs} documents")
    print(f"Cluster with the most documents contains: {max_docs} documents")

    output_data = {}
    for cluster_id in range(1000):
        if cluster_id in cluster_indices:
            sample_size = min(40, len(cluster_indices[cluster_id]))
            if sample_size > 0:
                sample_indices = random.sample(cluster_indices[cluster_id], sample_size)
                output_data[f"Cluster {cluster_id}"] = [documents[index] for index in sample_indices]

    # Writing the data to a JSON file
    with open('reddit_100k_100_1000_cluster_bm25.json', 'w') as file:
        json.dump(output_data, file, indent=4)

    # for cluster_id in range(20):
    #     if cluster_id in cluster_indices:
    #         print(f"Documents from Cluster {cluster_id}:")
    #         sample_size = min(10, len(cluster_indices[cluster_id]))
    #         if sample_size > 0:
    #             sample_indices = random.sample(cluster_indices[cluster_id], sample_size)
    #             for index in sample_indices:
    #                 print(documents[index])

def process_action(input_file, tokenizer, bm25_state_path, centroids, batch_size):
    # Load the BM25 state and prepare BM25 model
    state = load_tensors(bm25_state_path)
    document_input_ids = state['document_input_ids'].to(device)
    vocab_size = state['vocab_size']
    bm25 = TokenizedBM25(vocab_size=vocab_size)
    bm25.index(document_input_ids)
    print("Loading lines")
    # Read the entire file into memory
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Process the specified range of lines
    results = []
    for idx in tqdm(range(0, len(lines)), desc="Processing documents"):
        data = json.loads(lines[idx])
        test_documents = data["syms"]
        print("begin_tokenizing")
        test_input_ids, test_attention_masks = batch_tokenize_texts(tokenizer, test_documents, batch_size=batch_size)
        print("finished tokenization")
        test_input_ids = [pad_to_length(torch.tensor(t, device=device)) for t in test_input_ids]
        # print("Number of input ids:", len(test_input_ids))
        if not test_input_ids:
            print("test_input_ids is empty")
            print(test_input_ids)
        test_input_ids = torch.stack(test_input_ids).to(device)

        test_bag = bm25.docs_to_bags(test_input_ids).to_sparse_coo().float().to(device)
        result = torch.sparse.mm(test_bag, centroids.T.to(device)).to_dense()
        max_values, max_indices = torch.max(result, dim=1)

        action_types = [str(index.item()) for index in max_indices]
        data["cluster_type"] = action_types
        lines[idx] = json.dumps(data) + '\n'  # Update the line with new data

    # Write the updated contents back to the file
    with open(input_file, 'w') as file:
        file.writelines(lines)

    print("Processing completed and updated in", input_file)


def main(args):
    # Load dataset
    dataset = load_dataset('json', data_files=args.samples_filename, split='train')
    # cluster_assignments, centroids = cluster_dataset(
    #     dataset=dataset,
    #     model="gtr_base",
    #     query_key="document",
    #     document_key="document",
    #     query_to_doc=True,
    #     cluster_size=args.cluster_size,
    #     save_path = args.bm25_state_path
    # )

    # Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.model_max_length)

    # Tokenize documents in the dataset
    input_ids, attention_masks = batch_tokenize_texts(tokenizer, dataset['document'], batch_size=args.batch_size,max_length=args.model_max_length)
    dataset = dataset.add_column('input_ids', input_ids)
    dataset = dataset.add_column('attention_mask', attention_masks)

    # Clustering
    cluster_assignments, centroids = cluster_dataset(
        dataset=dataset,
        model="bm25",
        query_key='input_ids',
        document_key='input_ids',
        query_to_doc=True,
        cluster_size=args.cluster_size,
        save_path = args.bm25_state_path
    )
    

    cluster_assignments_path = f'{args.data_dir}/cluster_assignments.pkl'
    with open(cluster_assignments_path, 'wb') as f:
        pickle.dump(cluster_assignments, f)
    
    centroids_path = f'{args.data_dir}/centroids.pkl'
    with open(centroids_path, 'wb') as f:
        pickle.dump(centroids, f)

    with open(cluster_assignments_path, 'rb') as f:
        cluster_assignments = pickle.load(f)

    with open(centroids_path, 'rb') as f:
        centroids = pickle.load(f)

    # Process cluster results
    process_clusters(cluster_assignments, dataset['document'])

    # Load BM25 state
    #process_action(args.input_filename, tokenizer, args.bm25_state_path, centroids, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and cluster textual data.")
    parser.add_argument('--samples_filename', type=str, default='data/reddit/train_100000_samples_200.jsonl', help='Path to the samples dataset file.')
    parser.add_argument('--input_filename', type=str, default='data/reddit/train_processed_100.jsonl', help='Path to the input dataset file.')
    parser.add_argument('--model_path', type=str, default='Salesforce/SFR-Embedding-Mistral', help='Path to the model used for tokenization.')
    parser.add_argument('--model_max_length', type=int, default=128, help='Maximum length of the tokenized sequences.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size used for tokenization.')
    parser.add_argument('--cluster_size', type=int, default=100, help='Size of each cluster.')
    parser.add_argument('--bm25_state_path', type=str, default='data/reddit/bm25_state.pth', help='Path to the saved BM25 state file.')
    parser.add_argument('--data_dir', type=str, default='data/reddit', help='Base directory for saving and loading data files.')
    args = parser.parse_args()

    main(args)

#bert-base-uncased
#sentence-transformers/paraphrase-distilroberta-base-v1
#Salesforce/SFR-Embedding-Mistral