from hba import HonestyBasedAlgorithm
from kba import KLDBasedAlgorithm
from vba import VectorBasedAlgorithm
from scipy.stats import entropy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def prepare_data_for_anomaly_detection(df):
    """
    Prepare dataset for anomaly detection with enhanced feature extraction

    Args:
        df: Input DataFrame with 'text' and 'label' columns

    Returns:
        DataFrame with comprehensive features for anomaly detection
    """
    # Use sentence transformers for semantic embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for text
    text_embeddings = model.encode(df['text'].tolist())

    # One-hot encode labels
    label_mapping = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    label_names = [label_mapping[label] for label in df['label']]
    label_embeddings = model.encode(label_names)

    # Calculate advanced semantic distances
    semantic_distances = []
    cosine_similarities = []
    entropy_values = []

    for text_emb, label_emb in zip(text_embeddings, label_embeddings):
        # Cosine similarity and distance
        cosine_sim = np.dot(text_emb, label_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(label_emb))
        distance = 1 - cosine_sim
        semantic_distances.append(distance)
        cosine_similarities.append(cosine_sim)

        # Entropy of embedding
        hist, _ = np.histogram(text_emb, bins='auto', density=True)
        hist = hist + np.finfo(float).eps
        hist = hist / np.sum(hist)
        entropy_values.append(entropy(hist))

    # Enhanced features
    features = pd.DataFrame({
        'text_length': df['text'].str.len(),
        'word_count': df['text'].str.split().str.len(),
        'unique_word_ratio': df['text'].apply(lambda x: len(set(x.split())) / len(x.split())),
        'semantic_distance': semantic_distances,
        'cosine_similarity': cosine_similarities,
        'embedding_entropy': entropy_values,
        'punctuation_ratio': df['text'].apply(lambda x: sum(1 for c in x if c in '.,!?:;') / max(len(x), 1))
    })

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_features = tfidf.fit_transform(df['text']).toarray()
    tfidf_df = pd.DataFrame(
        tfidf_features,
        columns=[f'tfidf_feature_{i}' for i in range(tfidf_features.shape[1])]
    )

    # Combine all features
    final_features = pd.concat([features, tfidf_df], axis=1)

    # Optional: Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    final_features_scaled = pd.DataFrame(
        scaler.fit_transform(final_features),
        columns=final_features.columns,
        index=df.index
    )

    return final_features_scaled


def detect_mislabeled_texts(df, anomaly_data):
    """
    Enhanced mislabeled text detection with more robust analysis

    Args:
        df: Original DataFrame with labels
        anomaly_data: Prepared features for anomaly detection

    Returns:
        Dictionary of potentially mislabeled texts with enhanced insights
    """
    # Initialize algorithms with more sensitive thresholds
    hba = HonestyBasedAlgorithm(num_swarm_realizations=500, threshold_c2=0.2)
    kba = KLDBasedAlgorithm(num_swarm_realizations=500, threshold_c2=0.2)
    vba = VectorBasedAlgorithm(num_swarm_realizations=500, threshold_c1=0.2, threshold_c2=0.2)

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

        # Detailed label-wise analysis
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


def visualize_results(df, anomaly_data, results):
    """
    Visualization that highlights the proportion of normal vs anomalous data
    """
    # Identify anomalies
    intersection_indices = list(results['intersection_anomalies'])

    plt.figure(figsize=(20, 15))
    plt.suptitle('Anomaly Detection: Normal vs Anomalous Data', fontsize=16)

    # 1. Pie Chart of Anomalies
    plt.subplot(2, 2, 1)
    anomaly_counts = {
        'Normal Data': len(df) - len(intersection_indices),
        'High-Confidence Anomalies': len(intersection_indices)
    }
    plt.pie(
        anomaly_counts.values(),
        labels=anomaly_counts.keys(),
        autopct='%1.1f%%',
        colors=['lightgreen', 'salmon'],
        explode=(0, 0.1)
    )
    plt.title('Proportion of Normal vs Anomalous Data', fontsize=14)

    # 2. Stacked Bar of Normal vs Anomalous by Label
    plt.subplot(2, 2, 2)
    label_mapping = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

    # Prepare data for stacked bar
    normal_counts = df[~df.index.isin(intersection_indices)]['label'].value_counts()
    anomaly_counts = df.loc[intersection_indices, 'label'].value_counts()

    # Create DataFrame for plotting
    comparison_df = pd.DataFrame({
        'Normal': normal_counts,
        'Anomaly': anomaly_counts
    }).fillna(0)
    comparison_df.index = [label_mapping[idx] for idx in comparison_df.index]

    comparison_df.plot(kind='bar', stacked=True, ax=plt.gca(),
                       color=['lightgreen', 'salmon'])
    plt.title('Normal vs Anomalous Data by Label', fontsize=14)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.legend(title='Data Type')

    # 3. Density Plot of Semantic Features
    plt.subplot(2, 2, 3)
    plt.title('Semantic Distance Distribution', fontsize=14)

    # Separate normal and anomalous data
    normal_distances = anomaly_data.loc[~anomaly_data.index.isin(intersection_indices), 'semantic_distance']
    anomaly_distances = anomaly_data.loc[intersection_indices, 'semantic_distance']

    sns.histplot(
        normal_distances,
        label='Normal Data',
        color='lightgreen',
        kde=True,
        alpha=0.6
    )
    sns.histplot(
        anomaly_distances,
        label='Anomalies',
        color='salmon',
        kde=True,
        alpha=0.6
    )
    plt.xlabel('Semantic Distance')
    plt.ylabel('Density')
    plt.legend()

    # 4. Detailed Anomaly Metrics
    plt.subplot(2, 2, 4)
    metrics_df = pd.DataFrame({
        'Metric': ['Total Samples', 'Normal Samples', 'Anomaly Samples'],
        'Count': [
            len(df),
            len(df) - len(intersection_indices),
            len(intersection_indices)
        ]
    })
    sns.barplot(
        x='Metric',
        y='Count',
        data=metrics_df,
        palette=['blue', 'lightgreen', 'salmon']
    )
    plt.title('Data Sample Composition', fontsize=14)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print(f"Total Samples: {len(df)}")
    print(
        f"Normal Samples: {len(df) - len(intersection_indices)} ({(1 - len(intersection_indices) / len(df)) * 100:.2f}%)")
    print(f"Anomaly Samples: {len(intersection_indices)} ({len(intersection_indices) / len(df) * 100:.2f}%)")

    return results

def main():
    # Load dataset
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/wangrongsheng/ag_news/" + splits["test"])

    print(list(df.columns))
    print(df['label'].value_counts())

    # Prepare data for anomaly detection
    anomaly_data = prepare_data_for_anomaly_detection(df)



    # Detect potentially mislabeled texts
    results = detect_mislabeled_texts(df, anomaly_data)
    # Visualize results
    visualize_results(df, anomaly_data, results)

    # Optional: Export results for further investigation
#    for algo, candidates in results['mislabeled_candidates'].items():
#       candidates.to_csv(f'{algo}_mislabeled_candidates.csv', index=True)

if __name__ == "__main__":
    main()
