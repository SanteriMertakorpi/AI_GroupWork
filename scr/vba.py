import numpy as np
from scipy.stats import entropy
import pandas as pd

class VectorBasedAlgorithm:
    def __init__(self, num_swarm_realizations=400,threshold_c1=0.6, threshold_c2=0.6):
        """
        Initialize VBA algorithm parameters

        Args:
            num_swarm_realizations: Number of swarm formations to perform
            threshold_c1: Threshold for flagging meters within a swarm
            threshold_c2: Threshold for final anomaly decision
        """
        self.num_swarm_realization = num_swarm_realizations
        self.threshold_c1 = threshold_c1
        self.threshold_c2 = threshold_c2
    
    def form_random_swarm(self, meter_data, swarm_size_range=(5,10)):
        """
        Form a random swarm of smart meters

        Args:
            meter_data: Dataframe with meter readings
            swarm_size_range: Tuple of (min, max) swarm size
        
        Returns:
            Dataframe containing only the selected swarm meters
        """

        swarm_size = np.random.randint(swarm_size_range[0], swarm_size_range[1] )
        selected_meters = np.random.choice(meter_data.index, size=swarm_size, replace=False)
        return meter_data.loc[selected_meters]
    
    def calculate_metrics(self, readings):
        """
        Calculate mean and entropy for a set of readings

        Args:
            readings: Array of meter readings
        
        Returns:
            mean, entropy
        """
        mean = np.mean(readings)

        # Histogram and probability distribution
        hist, _ = np.histogram(readings, bins='auto',density=True)

        # Add constant to prevent log(0)
        hist = hist + np.finfo(float).eps
        hist = hist / np.sum(hist)

        entro = entropy(hist)

        return mean, entro
    
    def detect_anomalies(self, meter_data):
        """
        Detect anomalous meters with VBA

        Args:
            meter_data: Dataframe with meter readings (meters = rows, readings = columns)
        
        Returns:
            List of meter IDs flagged as anomalous
        """

        num_meters = len(meter_data)
        meter_flags = np.zeros((num_meters, self.num_swarm_realization))

        for j in range(self.num_swarm_realization):
            swarm = self.form_random_swarm(meter_data)

            average_consumption = swarm.mean()

            center_mean, center_entropy = self.calculate_metrics(average_consumption)

            deviations = []
            for idx, meter_readings in swarm.iterrows():
                meter_mean, meter_entropy = self.calculate_metrics(meter_readings)

                deviation = np.sqrt(
                    (center_mean - meter_mean)**2 +
                    (center_entropy - meter_entropy)**2
                )

                deviations.append((idx,deviation))
            
            max_deviation = max(dev for _, dev in deviations)
            for idx, deviation in deviations:
                if deviation > max_deviation * self.threshold_c1:
                    meter_flags[meter_data.index.get_loc(idx), j] =1
        flag_sums = np.sum(meter_flags, axis=1)
        max_flags = np.max(flag_sums)
        anomalous_meters = []

        for i in range (num_meters):
            if flag_sums[i] > max_flags * self.threshold_c2:
                anomalous_meters.append(meter_data.index[i])
        
        return anomalous_meters
