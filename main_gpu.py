import torch
import random
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from bm25_pt.bm25 import TokenizedBM25
from cluster import cluster_dataset, pad_to_length, embed_for_clustering
import json
import argparse
from datasets import Dataset

from embed import (
    DenseEncoder, 
    embed_with_cache,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
print(f"Using device: {device}") #


def load_tensors(path: str) -> dict:
    return torch.load(path, map_location=device)


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
    with open('reddit_100k_100_1000_output_gtr.json', 'w') as file:
        json.dump(output_data, file, indent=4)

    # for cluster_id in range(20):
    #     if cluster_id in cluster_indices:
    #         print(f"Documents from Cluster {cluster_id}:")
    #         sample_size = min(10, len(cluster_indices[cluster_id]))
    #         if sample_size > 0:
    #             sample_indices = random.sample(cluster_indices[cluster_id], sample_size)
    #             for index in sample_indices:
    #                 print(documents[index])


def process_action(input_file, kmeans):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    model_name = "sentence-transformers/gtr-t5-base"
    model = DenseEncoder(model_name, max_seq_length=256)

    batch_size = 200
    total_documents = len(lines)
    for batch_start in tqdm(range(0, total_documents, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_documents)
        batch_lines = lines[batch_start:batch_end]

        documents = [json.loads(line)['syms'] for line in batch_lines]
        all_syms = [sym for doc in documents for sym in doc]
        sym_dataset = Dataset.from_dict({'document': all_syms})

        embeddings = model.encode(sym_dataset, "document", batch_size=512)
        q = embeddings
        X = embeddings
        paired_vectors = torch.cat(
            [
                torch.cat((q, X), dim=0),
                torch.cat((X, q), dim=0),
            ], dim=1
        )
        paired_vectors /= paired_vectors.norm(dim=1, keepdim=True, p=2)
        paired_vectors = paired_vectors.cpu()
        queries = paired_vectors[:len(q)]
        # K-Means 聚类
        _distances, assignments = kmeans.index.search(queries, 1)
        action_types = [str(assignment[0]) for assignment in assignments]

        # 更新原始数据
        idx = 0
        update_idx = batch_start
        for line, doc in zip(batch_lines, documents):
            data = json.loads(line)
            doc_length = len(doc)
            data['action_type'] = action_types[idx:idx + doc_length]
            idx += doc_length
            lines[update_idx] = json.dumps(data) + '\n'
            update_idx += 1

    # 将更新的内容写回文件
    with open(input_file, 'w') as file:
        file.writelines(lines)

    print("Processing completed and updated in", input_file)


def main(args):
    # Load dataset
    dataset = load_dataset('json', data_files=args.samples_filename, split='train')
    cluster_assignments, centroids, kmeans = cluster_dataset(
        dataset=dataset,
        model="gtr_base",
        query_key="document",
        document_key="document",
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

    print("kmeans")
    print(kmeans.index)
    process_action(args.input_filename, kmeans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and cluster textual data.")
    parser.add_argument('--samples_filename', type=str, default='data/reddit/train_100000_samples_200.jsonl', help='Path to the samples dataset file.')
    parser.add_argument('--input_filename', type=str, default='data/reddit/train_processed_200.jsonl', help='Path to the input dataset file.')
    parser.add_argument('--model_path', type=str, default='Salesforce/SFR-Embedding-Mistral', help='Path to the model used for tokenization.')
    parser.add_argument('--model_max_length', type=int, default=128, help='Maximum length of the tokenized sequences.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size used for tokenization.')
    parser.add_argument('--cluster_size', type=int, default=200, help='Size of each cluster.')
    parser.add_argument('--bm25_state_path', type=str, default='data/reddit/bm25_state.pth', help='Path to the saved BM25 state file.')
    parser.add_argument('--data_dir', type=str, default='data/reddit', help='Base directory for saving and loading data files.')
    args = parser.parse_args()

    main(args)

#bert-base-uncased
#sentence-transformers/paraphrase-distilroberta-base-v1
#Salesforce/SFR-Embedding-Mistral