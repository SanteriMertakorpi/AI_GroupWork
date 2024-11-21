import numpy as np
import pandas as pd
from scipy.stats import entropy

class KLDBasedAlgorithm:
    def __init__(self, num_swarm_realizations=100, threshold_c2=0.6):
        """
        Initialize KBA algorithm parameters
        
        Args:
            num_swarm_realizations: Number of swarm formations to perform
            threshold_c2: Threshold for final anomaly decision
        """
        self.num_swarm_realizations = num_swarm_realizations
        self.threshold_c2 = threshold_c2
    
    def calculate_probability_distribution(self, readings, bins='auto'):
        """
        Calculate probability distribution from meter readings
        
        Args:
            readings: Array of meter readings
            bins: Number of bins or method for histogram
            
        Returns:
            Probability distribution
        """
        hist, bin_edges = np.histogram(readings, bins=bins, density=True)
        # Add small epsilon to avoid log(0)
        hist = hist + np.finfo(float).eps
        hist = hist / np.sum(hist)
        return hist
    
    def kullback_leibler_distance(self, p, q):
        """
        Calculate Kullback-Leibler distance between two distributions
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL distance
        """
        # Ensure distributions are the same length
        min_len = min(len(p), len(q))
        p, q = p[:min_len], q[:min_len]
        
        # Avoid log(0) by adding small epsilon
        p = p + np.finfo(float).eps
        q = q + np.finfo(float).eps
        
        # Normalize distributions
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL distance
        return np.sum(p * np.log(p / q))

    def detect_anomalies(self, meter_data, swarm_size_range=(5, 10)):
        """
        Detect anomalous meters using KBA

        Args:
            meter_data: DataFrame with meter readings (meters as rows, readings as columns)
            swarm_size_range: Tuple of (min, max) swarm size

        Returns:
            List of meter IDs flagged as anomalous
        """
        num_meters = len(meter_data)

        # Adjust swarm size range if it exceeds available meters
        swarm_size_range = (
            min(swarm_size_range[0], num_meters),
            min(swarm_size_range[1], num_meters)
        )

        meter_flags = np.zeros((num_meters, self.num_swarm_realizations))

        # Perform multiple swarm realizations
        for j in range(self.num_swarm_realizations):
            # Form random swarm
            swarm_size = np.random.randint(swarm_size_range[0], swarm_size_range[1])
            selected_meters = np.random.choice(meter_data.index, size=swarm_size, replace=False)
            swarm = meter_data.loc[selected_meters]

            # Calculate average consumption for swarm
            avg_consumption = swarm.mean()

            # Calculate probability distribution of average consumption
            avg_dist = self.calculate_probability_distribution(avg_consumption)

            # Calculate KL distances for each meter in swarm
            for meter_idx, meter_id in enumerate(selected_meters):
                # Calculate meter's individual distribution
                meter_dist = self.calculate_probability_distribution(swarm.iloc[meter_idx])

                # Calculate KL distance
                kl_distance = self.kullback_leibler_distance(meter_dist, avg_dist)

                # Flag meters with large KL distances
                if kl_distance > 1.0:  # Threshold can be adjusted
                    meter_flags[meter_data.index.get_loc(meter_id), j] = 1

        # Make final anomaly decisions
        flag_sums = np.sum(meter_flags, axis=1)
        max_flags = np.max(flag_sums)
        anomalous_meters = []

        for i in range(num_meters):
            if flag_sums[i] > max_flags * self.threshold_c2:
                anomalous_meters.append(meter_data.index[i])

        return anomalous_meters