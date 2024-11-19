import numpy as np
import pandas as pd
from scipy.linalg import lu

class HonestyBasedAlgorithm:
    def __init__(self, num_swarm_realizations=200, threshold_c2=0.6):
        """
        Initialize HBA algorithm parameters

        Args:
            num_swarm_realizations: Number of swarm formations to perform
            threshold_c2: Threshold for final anomaly decision
        """
        self.num_swarmn_realizations = num_swarm_realizations
        self.threshold_c2 = threshold_c2
    
    def calculate_virtual_collector(self, swarm_data, total_consumption):
        """
        Calculate virtual collector readings for the swarm

        Args:
            swarm_data: DataFrame containing swarm meter readings
            total_consumption: Total consumption for all meters

        Returns:
            Array of virtual collector readings
        """
        num_meters_total = len(total_consumption)
        num_meters_swarm = len(swarm_data)

        # Calculate approx total consumption for swarm using equation (10) from the paper
        virtual_collector = np.sum(swarm_data, axis=0) * (num_meters_total / num_meters_swarm)
        return virtual_collector
    
    def perform_lu_decomposition(self, P):
        """
        Perform LU decomposition on consumption matrix P

        Args:
            P: Consumption matrix
        
        Returns:
            L, U matrices
        """

        # Perform LU decomposition
        P = np.array(P)
        lu_matrix, piv = lu(P, permute_l=True)

        # Extract L and U matrices
        n = P.shape[0]
        L = lu_matrix[:, :n]
        U = lu_matrix[:, n:]

        return L, U
    
    def calculate_honesty_coefficients(self, swarm_data, virtual_collector):
        """
        Calculate honesty coefficients for meters in swarm

        Args:
            swarm_data: DataFrame containing swarm meter readings
            virtual_collector: Virtual collector readings
        
        Returns:
            Array of honesty coefficinets
        """
        P = np.array(swarm_data.T) # Transpose to match matrix format in the paper

        # Perform LU decomposition
        L, U = self.perform_lu_decomposition(P)

        # Solve Ly = P (equation 13)
        y = np.linalg.solve(L, virtual_collector)

        # Solve Uk = y (equation 14)
        k = np.linalg.solve(U,y)

        return k
    
    def detect_anomalies(self, meter_data, swarm_size_range=(5,10)):
        """
        Detect anomalous meters using HBA

        Args:
            meter_data: DataFrame with meter readings (meters as rows, readings as columns)
            swarm_size_range: Tuple of (min, max) swarm size
        
        Returns:
            List of meter IDs flagged as anomalous
        """

        num_meters = len(meter_data)
        meter_flags = np.zeros((num_meters, self.num_swarmn_realizations))

        # Calculate total consumption for virtual collector calucaltions
        total_consumption = meter_data.sum()

        # Perform multiple swarm realizations
        for j in range(self.num_swarmn_realizations):
            # Form random swarm
            swarm_size = np.random.randint(swarm_size_range[0], swarm_size_range[1])
            selected_meters = np.random.choice(meter_data.index, size=swarm_size, replace=False)
            swarm = meter_data.loc[selected_meters]

            # Calculate virtual collector readings
            virtual_collector = self.calculate_virtual_collector(swarm, total_consumption)

            # Calculate honesty coefficients
            try:
                honesty_coefficients = self.calculate_honesty_coefficients(swarm, virtual_collector)

                # Flag meters with suspicious honesty coefficients
                # Values significantlu different from 1 indicate potential anomalies
                for i, meter_id in enumerate(selected_meters):
                    if abs(honesty_coefficients[i] - 1.0) > 0.3: # Threshold can be adjusted
                        meter_flags[meter_data.index.get_loc(meter_id), j] = 1
            except np.linalg.LinAlgError:
                # Skip this swarm if matrix is singular
                continue

            # Final anomaly decisions
            flag_sums = np.sum(meter_flags, axis=1)
            max_flags = np.max(flag_sums)
            anomalous_meters = []

            for i in range(num_meters):
                if flag_sums[i] > max_flags * self.threshold_c2:
                    anomalous_meters.append(meter_data.index[i])
            
            return anomalous_meters
    