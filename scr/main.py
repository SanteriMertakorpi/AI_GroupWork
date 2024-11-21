import pandas as pd
import numpy as np
from hba import HonestyBasedAlgorithm
from kba import KLDBasedAlgorithm
from vba import VectorBasedAlgorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch


def prepare_data_for_anomaly_detection(df):
    """
    Prepare dataset for anomaly detection by creating features 
    that capture the semantic relationship between text and label

    Args:
        df: Input DataFrame with 'text' and 'label' columns

    Returns:
        DataFrame with features for anomaly detection
    """
    # Use sentence transformers for semantic embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for text
    text_embeddings = model.encode(df['text'].tolist())
    
    # One-hot encode labels
    label_mapping = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    label_names = [label_mapping[label] for label in df['label']]
    label_embeddings = model.encode(label_names)
    
    # Calculate semantic distance between text and label
    semantic_distances = []
    for text_emb, label_emb in zip(text_embeddings, label_embeddings):
        # Cosine similarity (lower value indicates less semantic alignment)
        distance = 1 - np.dot(text_emb, label_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(label_emb))
        semantic_distances.append(distance)
    
    # Additional features
    features = pd.DataFrame({
        'text_length': df['text'].str.len(),
        'word_count': df['text'].str.split().str.len(),
        'semantic_distance': semantic_distances
    })
    # Combine all features
    # Optional: Add TF-IDF features for additional context
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_features = tfidf.fit_transform(df['text']).toarray()
    tfidf_df = pd.DataFrame(
        tfidf_features, 
        columns=[f'tfidf_feature_{i}' for i in range(tfidf_features.shape[1])]
    )
    
    # Combine all features
    final_features = pd.concat([features, tfidf_df], axis=1)
    
    return final_features

def detect_mislabeled_texts(df, anomaly_data):
    """
    Detect potentially mislabeled texts using multiple algorithms
    
    Args:
        df: Original DataFrame with labels
        anomaly_data: Prepared features for anomaly detection
    
    Returns:
        Dictionary of potentially mislabeled texts
    """
    # Initialize algorithms
    hba = HonestyBasedAlgorithm(num_swarm_realizations=400, threshold_c2=0.5)
    kba = KLDBasedAlgorithm(num_swarm_realizations=400, threshold_c2=0.5)
    vba = VectorBasedAlgorithm(num_swarm_realizations=400, threshold_c1=0.5, threshold_c2=0.5)
    
    # Detect anomalies
    anomalies = {
        'HBA': hba.detect_anomalies(anomaly_data),
        'KBA': kba.detect_anomalies(anomaly_data),
        'VBA': vba.detect_anomalies(anomaly_data)
    }
    
    # Combine and analyze anomalies
    mislabeled_candidates = {}
    
    for algo, indices in anomalies.items():
        # Get the anomalous entries
        anomalous_entries = df.loc[indices]
        
        # Group anomalies by label
        label_anomalies = anomalous_entries.groupby('label').size()
        
        print(f"\n{algo} Anomalies by Label:")
        print(label_anomalies)
        
        # Store candidates for further investigation
        mislabeled_candidates[algo] = anomalous_entries
    
    # Find intersection of anomalies across algorithms
    intersection_anomalies = set(anomalies['HBA']) & set(anomalies['KBA']) & set(anomalies['VBA'])
    
    print("\nIntersection of Anomalies:")
    intersection_df = df.loc[list(intersection_anomalies)]
    print(intersection_df[['text', 'label']])
    
    # Detailed analysis of intersection
    if not intersection_anomalies:
        print("No consistent anomalies found across all algorithms.")
    
    return {
        'all_anomalies': anomalies,
        'intersection_anomalies': intersection_anomalies,
        'mislabeled_candidates': mislabeled_candidates
    }

def main():
    # Load dataset
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["test"])
    
    # Prepare data for anomaly detection
    anomaly_data = prepare_data_for_anomaly_detection(df)
    
    # Detect potentially mislabeled texts
    results = detect_mislabeled_texts(df, anomaly_data)
    
    # Optional: Export results for further investigation
    for algo, candidates in results['mislabeled_candidates'].items():
        candidates.to_csv(f'{algo}_mislabeled_candidates.csv', index=True)

if __name__ == "__main__":
    main()