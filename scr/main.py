import pandas as pd
import numpy as np
from hba import HonestyBasedAlgorithm
from kba import KLDBasedAlgorithm
from vba import VectorBasedAlgorithm


def prepare_data_for_anomaly_detection(df):
    """
    Prepare dataset for anomaly detection by converting text features

    Args:
        df: Input DataFrame

    Returns:
        Processed DataFrame suitable for anomaly detection
    """
    # Convert text features to numerical representations
    # Use text length and word count as basic features
    df['title_length'] = df['text'].str.len()
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()

    # Select numerical features for anomaly detection
    anomaly_features = ['title_length', 'text_length', 'word_count']

    # Create a new DataFrame with index preserved
    return df[anomaly_features].reset_index(drop=True)


def detect_anomalies_with_multiple_algorithms(data):
    """
    Apply multiple anomaly detection algorithms

    Args:
        data: Prepared numerical DataFrame

    Returns:
        Dictionary of anomalous indices for each algorithm
    """
    # Initialize algorithms
    hba = HonestyBasedAlgorithm()
    kba = KLDBasedAlgorithm()
    vba = VectorBasedAlgorithm()

    # Detect anomalies
    anomalies = {
        'HBA': hba.detect_anomalies(data),
        'KBA': kba.detect_anomalies(data),
        'VBA': vba.detect_anomalies(data)
    }

    return anomalies


def analyze_anomaly_results(df, anomalies):
    """
    Analyze and print anomaly detection results

    Args:
        df: Original DataFrame
        anomalies: Dictionary of anomalous indices from different algorithms
    """
    print("Anomaly Detection Results:")
    for algorithm, anomalous_indices in anomalies.items():
        print(f"\n{algorithm} Anomalies:")

        # Print anomalous entries details
        anomalous_entries = df.iloc[anomalous_indices]
        print(f"Number of anomalies detected: {len(anomalous_indices)}")

        # Display first few anomalous entries
        print("\nSample Anomalous Entries:")
        print(anomalous_entries[['text']].head())


def main():
    # Synthetic data generation since we don't have actual dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'text': [
            'This is a normal text about technology',
            'Another regular news article discussing politics',
            'Extremely long text with repetitive words ' * 10,  # Potential anomaly
            'Short text',
            'Normal length text discussing current events',
            'Very unusual text with strange word patterns ' * 5,  # Potential anomaly
        ]
    })

    # Prepare data for anomaly detection
    anomaly_data = prepare_data_for_anomaly_detection(df)

    # Modify each algorithm to use a smaller swarm size range
    hba = HonestyBasedAlgorithm()
    kba = KLDBasedAlgorithm()
    vba = VectorBasedAlgorithm()

    # Modify VBA to use a smaller swarm size range
    vba_anomalies = vba.detect_anomalies(anomaly_data, swarm_size_range=(2, 5))

    # Analyze results
    anomalies = {
        'HBA': hba.detect_anomalies(anomaly_data, swarm_size_range=(2, 5)),
        'KBA': kba.detect_anomalies(anomaly_data, swarm_size_range=(2, 5)),
        'VBA': vba_anomalies
    }
    analyze_anomaly_results(df, anomalies)


if __name__ == "__main__":
    main()