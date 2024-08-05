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
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #
print(f"Using device: {device}") #


def load_tensors(path: str) -> dict:
    return torch.load(path, map_location=device)


def batch_tokenize_texts(tokenizer, texts, batch_size=8, max_length=128):
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


def create_graph(input_file, tokenizer, bm25_state_path, centroids, cluster_assignments, n_components=2):
    # Load the BM25 state and prepare BM25 model
    state = load_tensors(bm25_state_path)
    document_input_ids = state['document_input_ids'].to(device)
    query_input_ids= state['query_input_ids'].to(device)
    vocab_size = state['vocab_size']
    bm25 = TokenizedBM25(vocab_size=vocab_size)
    bm25.index(document_input_ids)
    queries = bm25.docs_to_bags(query_input_ids).to_sparse_coo()
    queries_dense = queries.to_dense().cpu().numpy()  # Convert sparse matrix to dense numpy array
    
    # Dimensionality reduction using PCA
    pca = PCA(n_components=n_components)
    queries_transformed = pca.fit_transform(queries_dense)  # Transform queries
    centroids_dense = centroids.to_dense().cpu().numpy()  # Assuming centroids are in tensor form
    centroids_transformed = pca.transform(centroids_dense)  # Transform centroids using the same PCA model
    
    # Prepare cluster color mapping
    cluster_colors = {cid: idx for idx, cid in enumerate(sorted(set(cluster_assignments.values())))}
    
    return queries_transformed, centroids_transformed, cluster_assignments, cluster_colors

def simplify_cluster_assignments(cluster_assignments):
    # Convert each list to a single integer directly mapping the document to its cluster
    simplified_assignments = {doc_id: clusters[0] for doc_id, clusters in cluster_assignments.items()}
    return simplified_assignments

def visualize_clusters(queries, centroids, cluster_assignments, n_components=2):
    # Assuming cluster_assignments now directly maps ids to cluster ids
    unique_clusters = set(cluster_assignments.values())
    cluster_colors = {cid: plt.cm.tab20(i/len(unique_clusters)) for i, cid in enumerate(unique_clusters)}
    
    fig, ax = plt.subplots()
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for cluster_id in unique_clusters:
            cluster_points = queries[[i for i in range(len(queries)) if cluster_assignments[i] == cluster_id], :]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=cluster_colors[cluster_id], label=f'Cluster {cluster_id}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    else:
        for cluster_id in unique_clusters:
            cluster_points = queries[[i for i in range(len(queries)) if cluster_assignments[i] == cluster_id], :]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[cluster_id], label=f'Cluster {cluster_id}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    ax.legend(title="Cluster IDs")
    plt.title(f'Cluster Visualization in {n_components}D Space')
    plt.show()

# Streamlit visualization function
def st_visualization(queries, centroids, cluster_assignments, cluster_colors):
    st.title("Query-Centroid Relationships Visualization")
    # dimension = st.selectbox("Select the number of dimensions for PCA:", [2, 3])
    fig = visualize_clusters(queries, centroids, cluster_assignments, cluster_colors, n_components=2)
    st.pyplot(fig)


def main(args):
    # Load dataset
    dataset = load_dataset('json', data_files=args.samples_filename, split='train')

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
    )
    cluster_assignments = simplify_cluster_assignments(cluster_assignments)
    # print("cluster_assignments")
    # print(cluster_assignments)
    # print(centroids.shape)
    st.write("cluster_assignments")
    st.write(cluster_assignments)
    st.write("Centroids Shape:", centroids.shape)
    with open('centroids.pkl', 'wb') as f:
        pickle.dump(centroids, f)

    with open('centroids.pkl', 'rb') as f:
        centroids = pickle.load(f)


    # Load BM25 state
    # create_graph(args.input_filename, tokenizer, args.bm25_state_path, centroids)
    queries, centroids, cluster_assignments, cluster_colors = create_graph(
        args.input_filename,
        tokenizer,
        args.bm25_state_path,
        centroids,
        cluster_assignments,
        n_components=args.pca_components  # Ensure this arg is parsed in argparse
    )
    st.write("Queries Shape:", queries.shape)
    st.write("Centroids Shape:", centroids.shape)
    st.write("Colors:", cluster_colors)

    st_visualization(queries, centroids, cluster_assignments, cluster_colors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and cluster textual data.")
    parser.add_argument('--samples_filename', type=str, default='data/reddit/train_100k_samples.jsonl', help='Path to the samples dataset file.')
    parser.add_argument('--input_filename', type=str, default='data/reddit/train_processed.jsonl', help='Path to the input dataset file.')
    parser.add_argument('--model_path', type=str, default='sentence-transformers/paraphrase-distilroberta-base-v1', help='Path to the model used for tokenization.')
    parser.add_argument('--model_max_length', type=int, default=128, help='Maximum length of the tokenized sequences.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size used for tokenization.')
    parser.add_argument('--cluster_size', type=int, default=1000, help='Size of each cluster.')
    parser.add_argument('--bm25_state_path', type=str, default='bm25_state.pth', help='Path to the saved BM25 state file.')
    parser.add_argument('--pca_components', type=int, default=2, help='Number of PCA components for visualization.')
    args = parser.parse_args()

    main(args)
